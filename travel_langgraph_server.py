import os
import json
import logging
from datetime import datetime, timedelta
from typing import TypedDict, Optional, Any

import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END


# =========================
# SETUP
# =========================

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("travel-langgraph")

app = FastAPI(title="Travel LangGraph Agent", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SNOWFLAKE_CONFIG = {
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE", "TRAVEL_RAG"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
}

FLIGHTS_FILE = os.getenv("FLIGHTS_FILE", "flights_cleaned.csv")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0,
    num_ctx=8192,
)


# =========================
# API MODELS
# =========================

class ChatRequest(BaseModel):
    message: str = Field(...)


class ChatResponse(BaseModel):
    answer: str
    debug: Optional[dict[str, Any]] = None


# =========================
# STATE
# =========================

class TravelState(TypedDict, total=False):
    user_message: str
    intent: str

    origin: Optional[str]
    destination: Optional[str]
    city: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    date: Optional[str]
    budget: Optional[float]
    max_price: Optional[float]
    preference: Optional[str]

    days: int
    dates: list[str]
    budget_split: dict[str, float]

    flights: Any
    hotels: list[dict[str, Any]]
    restaurants: list[dict[str, Any]]
    attractions: list[dict[str, Any]]
    rag_results: list[dict[str, Any]]

    answer: str


# =========================
# HELPERS
# =========================

def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def calculate_dates(start_date: str, end_date: str) -> tuple[int, list[str]]:
    start = parse_date(start_date)
    end = parse_date(end_date)

    if end < start:
        raise ValueError("end_date must be after or equal to start_date")

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    return len(dates), dates


def split_budget(total_budget: float, days: int) -> dict[str, float]:
    flight_budget = total_budget * 0.15
    hotel_budget_total = total_budget * 0.45
    food_budget_total = total_budget * 0.25
    activity_budget_total = total_budget * 0.15

    return {
        "flight_budget": round(flight_budget, 2),
        "hotel_budget_total": round(hotel_budget_total, 2),
        "hotel_budget_per_night": round(hotel_budget_total / max(days, 1), 2),
        "food_budget_total": round(food_budget_total, 2),
        "food_budget_per_day": round(food_budget_total / max(days, 1), 2),
        "activity_budget_total": round(activity_budget_total, 2),
    }


def sql_escape(value: str) -> str:
    return value.replace("'", "''")


def get_connection():
    return snowflake.connector.connect(**SNOWFLAKE_CONFIG)


def fetch_rows(query: str) -> list[tuple]:
    conn = get_connection()
    cur = conn.cursor()
    try:
        logger.info("Executing Snowflake query")
        cur.execute(query)
        return cur.fetchall()
    finally:
        cur.close()
        conn.close()


# =========================
# FLIGHTS DATA
# =========================

def load_flights_df() -> pd.DataFrame:
    df = pd.read_csv(FLIGHTS_FILE, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    required_cols = [
        "Flight Number",
        "Price",
        "DepTime",
        "ArrTime",
        "ActualElapsedTime",
        "FlightDate",
        "OriginCityName",
        "DestCityName",
        "Distance",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing flight columns: {missing}")

    df["Flight Number"] = df["Flight Number"].astype(str).str.strip()
    df["OriginCityName"] = df["OriginCityName"].astype(str).str.strip()
    df["DestCityName"] = df["DestCityName"].astype(str).str.strip()
    df["DepTime"] = df["DepTime"].astype(str).str.strip()
    df["ArrTime"] = df["ArrTime"].astype(str).str.strip()
    df["ActualElapsedTime"] = df["ActualElapsedTime"].astype(str).str.strip()

    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")

    df["FlightDate"] = pd.to_datetime(
        df["FlightDate"].astype(str).str.strip(),
        format="%Y-%m-%d",
        errors="coerce"
    )

    df = df.dropna(subset=["FlightDate", "Price"])

    return df


FLIGHTS_DF = load_flights_df()


def format_flight_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = []

    for _, row in df.iterrows():
        records.append({
            "flight_number": row["Flight Number"],
            "price": float(row["Price"]),
            "departure_time": row["DepTime"],
            "arrival_time": row["ArrTime"],
            "duration": row["ActualElapsedTime"],
            "flight_date": row["FlightDate"].strftime("%Y-%m-%d"),
            "origin_city": row["OriginCityName"],
            "destination_city": row["DestCityName"],
            "distance": float(row["Distance"]) if pd.notna(row["Distance"]) else None,
        })

    return records


def search_flights_data(
    origin: str,
    destination: str,
    start_date: str,
    budget: float,
    limit: int = 5
) -> dict[str, Any]:

    target_date = pd.to_datetime(start_date)

    route_df = FLIGHTS_DF[
        FLIGHTS_DF["OriginCityName"].str.contains(origin, case=False, na=False) &
        FLIGHTS_DF["DestCityName"].str.contains(destination, case=False, na=False)
    ].copy()

    if route_df.empty:
        return {
            "message": "No flights available for this route.",
            "flights": []
        }

    exact = route_df[route_df["FlightDate"] == target_date].copy()

    if not exact.empty:
        within_budget = exact[exact["Price"] <= budget]

        if not within_budget.empty:
            exact = within_budget

        exact = exact.sort_values("Price").head(limit)

        return {
            "message": "Exact-date flights found.",
            "flights": format_flight_records(exact)
        }

    route_df["date_diff"] = (route_df["FlightDate"] - target_date).abs().dt.days
    nearest = route_df[route_df["date_diff"] <= 3].copy()

    if nearest.empty:
        nearest = route_df.sort_values(["date_diff", "Price"]).head(limit)

        return {
            "message": "No exact or ±3 day flight found. Showing nearest available route options.",
            "flights": format_flight_records(nearest)
        }

    within_budget = nearest[nearest["Price"] <= budget]

    if not within_budget.empty:
        nearest = within_budget

    nearest = nearest.sort_values(["date_diff", "Price"]).head(limit)

    return {
        "message": "No exact-date flight found. Showing nearest available flights.",
        "flights": format_flight_records(nearest)
    }


# =========================
# SNOWFLAKE DATA FUNCTIONS
# =========================

def search_hotels_data(city: str, max_price_per_night: float, limit: int = 5):
    city_esc = sql_escape(city)

    query = f"""
        SELECT
            source_file,name, city, price, rating, room_type, max_occupancy, min_nights, text
        FROM travel_embeddings
        WHERE type = 'hotel'
          AND city ILIKE '%{city_esc}%'
          AND price IS NOT NULL
          AND price <= {max_price_per_night}
        ORDER BY price ASC NULLS LAST, rating DESC
        LIMIT {limit}
    """

    rows = fetch_rows(query)

    if not rows:
        query = f"""
            SELECT
                source_file,name, city, price, rating, room_type, max_occupancy, min_nights, text
            FROM travel_embeddings
            WHERE type = 'hotel'
              AND city ILIKE '%{city_esc}%'
              AND price IS NOT NULL
            ORDER BY price ASC NULLS LAST, rating DESC
            LIMIT {limit}
        """
        rows = fetch_rows(query)

    keys = [
        "source_file",
        "name",
        "city",
        "price_per_night",
        "rating",
        "room_type",
        "max_occupancy",
        "min_nights",
        "text",
    ]

    return [dict(zip(keys, row)) for row in rows]


def search_restaurants_data(city: str, max_average_cost: float, limit: int = 8):
    city_esc = sql_escape(city)

    query = f"""
        SELECT
            source_file,name, city, average_cost, rating, cuisines, text
        FROM travel_embeddings
        WHERE type = 'restaurant'
          AND city ILIKE '%{city_esc}%'
          AND average_cost IS NOT NULL
          AND average_cost <= {max_average_cost}
        ORDER BY rating DESC, average_cost ASC NULLS LAST
        LIMIT {limit}
    """

    rows = fetch_rows(query)

    if not rows:
        query = f"""
            SELECT
                source_file,name, city, average_cost, rating, cuisines, text
            FROM travel_embeddings
            WHERE type = 'restaurant'
              AND city ILIKE '%{city_esc}%'
              AND average_cost IS NOT NULL
            ORDER BY rating DESC, average_cost ASC NULLS LAST
            LIMIT {limit}
        """
        rows = fetch_rows(query)

    keys = ["source_file","name", "city", "average_cost", "rating", "cuisines", "text"]

    return [dict(zip(keys, row)) for row in rows]


def search_attractions_data(city: str, preference: str, limit: int = 8):
    city_esc = sql_escape(city)
    pref_esc = sql_escape(f"{preference} attractions in {city}")

    query = f"""
        SELECT
            source_file,name, city, address, phone_number, website, rating, text
        FROM travel_embeddings
        WHERE type = 'attraction'
          AND city ILIKE '%{city_esc}%'
        ORDER BY VECTOR_L2_DISTANCE(
            SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', '{pref_esc}'),
            embedding
        )
        LIMIT {limit}
    """

    rows = fetch_rows(query)

    keys = ["source_file","name", "city", "address", "phone_number", "website", "rating", "text"]

    return [dict(zip(keys, row)) for row in rows]


def general_rag_search(query_text: str, city: Optional[str] = None, limit: int = 8):
    filters = []

    if city:
        filters.append(f"city ILIKE '%{sql_escape(city)}%'")

    where_clause = ""
    if filters:
        where_clause = "WHERE " + " AND ".join(filters)

    q_esc = sql_escape(query_text)

    query = f"""
        SELECT
            name, city, type, text
        FROM travel_embeddings
        {where_clause}
        ORDER BY VECTOR_L2_DISTANCE(
            SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', '{q_esc}'),
            embedding
        )
        LIMIT {limit}
    """

    rows = fetch_rows(query)

    keys = ["name", "city", "type", "text"]

    return [dict(zip(keys, row)) for row in rows]


# =========================
# LANGGRAPH NODES
# =========================

def classify_intent_node(state: TravelState) -> TravelState:
    message = state["user_message"]

    prompt = f"""
Classify the user request.

Return ONLY valid JSON.

Possible intents:
- itinerary
- flights
- hotels
- restaurants
- attractions
- general_rag

Extract available fields:
origin, destination, city, start_date, end_date, date, budget, max_price, preference

Also extract:
- cuisine (example: seafood, italian, chinese, pizza, indian, etc)
- If no cuisine is mentioned, return null

Rules:
- Use null if missing.
- Dates must be YYYY-MM-DD.
- budget and max_price must be numbers or null.
- For restaurant/hotel/attraction requests, city should be destination/city.
- Do not explain.

User message:
{message}
"""

    response = llm.invoke(prompt).content.strip()

    # CLEAN markdown if present
    if response.startswith("```"):
        response = response.replace("```json", "").replace("```", "").strip()
    
    #  SAFE PARSE
    try:
        data = json.loads(response)
    except Exception:
        print("JSON PARSE FAILED:", response)

        # fallback empty data
        data = {}    
    #safe intent extraction
    intent = data.get("intent", "general_rag") 

    #  FORCE itinerary detection
    if (
        data.get("destination") and
        data.get("start_date") and
        data.get("end_date") and
        data.get("budget")
    ):
        intent = "itinerary"

    print("MODEL INTENT:", data.get("intent"))
    print("FINAL INTENT:", intent)



    return {
        "intent": intent,
        "origin": data.get("origin"),
        "destination": data.get("destination"),
        "city": data.get("city") or data.get("destination"),
        "start_date": data.get("start_date"),
        "end_date": data.get("end_date"),
        "date": data.get("date") or data.get("start_date"),
        "budget": data.get("budget"),
        "max_price": data.get("max_price") or data.get("budget"),
        "preference": data.get("preference") or "best",
        "cuisine": data.get("cuisine"),
    }



def route_by_intent(state: TravelState) -> str:
    intent = state.get("intent", "general_rag")

    if intent == "itinerary":
        return "constraints"
    if intent == "flights":
        return "flights_only"
    if intent == "hotels":
        return "hotels_only"
    if intent == "restaurants":
        return "restaurants_only"
    if intent == "attractions":
        return "attractions_only"

    return "general_rag"


def constraints_node(state: TravelState) -> TravelState:
    days, dates = calculate_dates(state["start_date"], state["end_date"])
    budget_split = split_budget(float(state["budget"]), days)

    return {
        "days": days,
        "dates": dates,
        "budget_split": budget_split,
    }


def flights_node(state: TravelState) -> TravelState:
    origin = state.get("origin")

    if not origin:
        return {
            "flights": {
                "message": "Origin not provided, flight search skipped.",
                "flights": []
            }
        }

    flights = search_flights_data(
        origin=origin,
        destination=state["destination"],
        start_date=state["start_date"],
        budget=state["budget_split"]["flight_budget"],
        limit=5,
    )

    return {"flights": flights}


def hotels_node(state: TravelState) -> TravelState:
    hotels = search_hotels_data(
        city=state["destination"],
        max_price_per_night=state["budget_split"]["hotel_budget_per_night"],
        limit=5,
    )

    return {"hotels": hotels}


def restaurants_node(state: TravelState) -> TravelState:
    restaurants = search_restaurants_data(
        city=state["destination"],
        max_average_cost=state["budget_split"]["food_budget_per_day"],
        limit=8,
    )

    return {"restaurants": restaurants}


def attractions_node(state: TravelState) -> TravelState:
    attractions = search_attractions_data(
        city=state["destination"],
        preference=state.get("preference") or "best",
        limit=8,
    )

    return {"attractions": attractions}


def generate_answer_node(state: TravelState) -> TravelState:
    prompt = f"""
You are a travel planner.

Use ONLY the provided data.
Do not invent flights, hotels, restaurants, attractions, prices, flight numbers, or addresses.
If something is missing, say "Not found".
At the end, include a "Sources Used" section listing the source_file for hotels, restaurants, and attractions used.

Trip:
- Origin: {state.get("origin")}
- Destination: {state["destination"]}
- Start date: {state["start_date"]}
- End date: {state["end_date"]}
- Dates: {state["dates"]}
- Days: {state["days"]}
- Total budget: {state["budget"]}
- Budget split: {state["budget_split"]}

Flights:
{json.dumps(state.get("flights"), indent=2)}

Hotels:
{json.dumps(state.get("hotels"), indent=2)}

Restaurants:
{json.dumps(state.get("restaurants"), indent=2)}

Attractions:
{json.dumps(state.get("attractions"), indent=2)}

Rules:
- Always create the itinerary for the original requested dates.
- If nearest flights are shown, mention them as alternatives but do not shift the trip dates automatically.
- Choose one hotel from the provided hotel list.
- Use restaurants and attractions only from the provided lists.
- Do not use hotels with minimum stay greater than the trip days if a better option exists.
- Keep the output concise.

Output format:

Flight Details:
...

Recommended Hotel:
...

Day 1 (date):
- Morning:
- Lunch:
- Afternoon:
- Evening:

Day 2:
...

Total Cost Summary:
...
"""

    answer = llm.invoke(prompt).content

    return {"answer": answer}


# =========================
# SPECIFIC INTENT NODES
# =========================

def flights_only_node(state: TravelState) -> TravelState:
    if not state.get("origin") or not state.get("destination") or not state.get("date"):
        return {
            "flights": {
                "message": "Missing origin, destination, or date for flight search.",
                "flights": []
            }
        }

    flights = search_flights_data(
        origin=state["origin"],
        destination=state["destination"],
        start_date=state["date"],
        budget=float(state["max_price"]) if state.get("max_price") else 999999,
        limit=5,
    )

    return {"flights": flights}


def hotels_only_node(state: TravelState) -> TravelState:
    city = state.get("city")

    if not city:
        return {"hotels": []}

    hotels = search_hotels_data(
        city=city,
        max_price_per_night=float(state["max_price"]) if state.get("max_price") else 999999,
        limit=5,
    )

    return {"hotels": hotels}


def restaurants_only_node(state: TravelState) -> TravelState:
    city = state.get("city")

    if not city:
        return {"restaurants": []}

    restaurants = search_restaurants_data(
        city=city,
        max_average_cost=float(state["max_price"]) if state.get("max_price") else 999999,
        limit=8,
    )
    cuisine = (state.get("cuisine") or "").lower()

    if cuisine:
        restaurants = [
            r for r in restaurants
            if r.get("cuisines") and cuisine in r.get("cuisines").lower()
        ]
    return {"restaurants": restaurants}


def attractions_only_node(state: TravelState) -> TravelState:
    city = state.get("city")

    if not city:
        return {"attractions": []}

    attractions = search_attractions_data(
        city=city,
        preference=state.get("preference") or state["user_message"],
        limit=8,
    )

    return {"attractions": attractions}


def general_rag_node(state: TravelState) -> TravelState:
    results = general_rag_search(
        query_text=state["user_message"],
        city=state.get("city"),
        limit=8,
    )

    return {"rag_results": results}


# =========================
# ANSWER NODES
# =========================

def answer_flights_node(state: TravelState) -> TravelState:
    prompt = f"""
Answer the user's flight question using ONLY this data.

User question:
{state['user_message']}

Flights:
{json.dumps(state.get('flights'), indent=2)}

Do not invent flights.
"""

    return {"answer": llm.invoke(prompt).content}


# def answer_hotels_node(state: TravelState) -> TravelState:
#     prompt = f"""
# Answer the user's hotel question using ONLY this data.

# User question:
# {state['user_message']}

# Hotels:
# {json.dumps(state.get('hotels'), indent=2)}


# Rules:
# - Do not invent hotels.
# - List ALL hotels provided in the data.
# - Include name, city, price per night, room type, max occupancy, rating, and minimum nights if available.
# - If the user asks for cheap hotels, sort/explain from cheapest to costliest.

# Answer in a clean numbered list.
# """

#     return {"answer": llm.invoke(prompt).content}

def answer_hotels_node(state: TravelState) -> TravelState:
    hotels = state.get("hotels", [])

    if not hotels:
        return {"answer": "No hotels found for your request."}

    lines = ["Here are the hotels found:\n"]

    for i, h in enumerate(hotels, start=1):
        lines.append(
            f"{i}. {h.get('name')}\n"
            f"   - City: {h.get('city')}\n"
            f"   - Price per night: ${h.get('price_per_night')}\n"
            f"   - Room type: {h.get('room_type')}\n"
            f"   - Maximum occupancy: {h.get('max_occupancy')}\n"
            f"   - Rating: {h.get('rating')}\n"
            f"   - Minimum nights: {h.get('min_nights')}\n"
        )

    return {"answer": "\n".join(lines)}


# def answer_restaurants_node(state: TravelState) -> TravelState:
#     prompt = f"""
# Answer the user's restaurant question using ONLY this data.

# User question:
# {state['user_message']}

# Restaurants:
# {json.dumps(state.get('restaurants'), indent=2)}

# Do not invent restaurants.
# """

#     return {"answer": llm.invoke(prompt).content}
def answer_restaurants_node(state: TravelState) -> TravelState:
    restaurants = state.get("restaurants", [])

    if not restaurants:
        return {"answer": "No restaurants found for your request."}

    lines = ["Here are the restaurants found:\n"]

    for i, r in enumerate(restaurants, start=1):
        lines.append(
            f"{i}. {r.get('name')}\n"
            f"   - City: {r.get('city')}\n"
            f"   - Average cost: ${r.get('average_cost')}\n"
            f"   - Rating: {r.get('rating')}\n"
            f"   - Cuisines: {r.get('cuisines')}\n"
        )

    return {"answer": "\n".join(lines)}


# def answer_attractions_node(state: TravelState) -> TravelState:
#     prompt = f"""
# Answer the user's attraction question using ONLY this data.

# User question:
# {state['user_message']}

# Attractions:
# {json.dumps(state.get('attractions'), indent=2)}

# Do not invent attractions.
# """

#     return {"answer": llm.invoke(prompt).content}

def answer_attractions_node(state: TravelState) -> TravelState:
    attractions = state.get("attractions", [])

    if not attractions:
        return {"answer": "No attractions found for your request."}

    lines = ["Here are the attractions found:\n"]

    for i, a in enumerate(attractions, start=1):
        lines.append(
            f"{i}. {a.get('name')}\n"
            f"   - City: {a.get('city')}\n"
            f"   - Address: {a.get('address')}\n"
            f"   - Phone: {a.get('phone_number')}\n"
            f"   - Website: {a.get('website')}\n"
            f"   - Rating: {a.get('rating')}\n"
        )

    return {"answer": "\n".join(lines)}


def answer_general_rag_node(state: TravelState) -> TravelState:
    prompt = f"""
Answer the user question using ONLY the provided context.

User question:
{state['user_message']}

Context:
{json.dumps(state.get('rag_results'), indent=2)}

If not found, say "Not found in context."
"""

    return {"answer": llm.invoke(prompt).content}


# =========================
# BUILD LANGGRAPH
# =========================

graph_builder = StateGraph(TravelState)

graph_builder.add_node("classify_intent", classify_intent_node)

graph_builder.add_node("constraints", constraints_node)
graph_builder.add_node("flights", flights_node)
graph_builder.add_node("hotels", hotels_node)
graph_builder.add_node("restaurants", restaurants_node)
graph_builder.add_node("attractions", attractions_node)
graph_builder.add_node("generate_answer", generate_answer_node)

graph_builder.add_node("flights_only", flights_only_node)
graph_builder.add_node("hotels_only", hotels_only_node)
graph_builder.add_node("restaurants_only", restaurants_only_node)
graph_builder.add_node("attractions_only", attractions_only_node)
graph_builder.add_node("general_rag", general_rag_node)

graph_builder.add_node("answer_flights", answer_flights_node)
graph_builder.add_node("answer_hotels", answer_hotels_node)
graph_builder.add_node("answer_restaurants", answer_restaurants_node)
graph_builder.add_node("answer_attractions", answer_attractions_node)
graph_builder.add_node("answer_general_rag", answer_general_rag_node)

graph_builder.add_edge(START, "classify_intent")

graph_builder.add_conditional_edges(
    "classify_intent",
    route_by_intent,
    {
        "constraints": "constraints",
        "flights_only": "flights_only",
        "hotels_only": "hotels_only",
        "restaurants_only": "restaurants_only",
        "attractions_only": "attractions_only",
        "general_rag": "general_rag",
    }
)

# itinerary path
graph_builder.add_edge("constraints", "flights")
graph_builder.add_edge("flights", "hotels")
graph_builder.add_edge("hotels", "restaurants")
graph_builder.add_edge("restaurants", "attractions")
graph_builder.add_edge("attractions", "generate_answer")
graph_builder.add_edge("generate_answer", END)

# specific paths
graph_builder.add_edge("flights_only", "answer_flights")
graph_builder.add_edge("answer_flights", END)

graph_builder.add_edge("hotels_only", "answer_hotels")
graph_builder.add_edge("answer_hotels", END)

graph_builder.add_edge("restaurants_only", "answer_restaurants")
graph_builder.add_edge("answer_restaurants", END)

graph_builder.add_edge("attractions_only", "answer_attractions")
graph_builder.add_edge("answer_attractions", END)

graph_builder.add_edge("general_rag", "answer_general_rag")
graph_builder.add_edge("answer_general_rag", END)

travel_graph = graph_builder.compile()


# =========================
# API ROUTES
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        result = travel_graph.invoke({"user_message": req.message})
        #validation code 
        answer = result.get("answer", "")

        allowed_names = set()

        # collect valid names
        for h in result.get("hotels", []):
            allowed_names.add(str(h.get("name")))

        for r in result.get("restaurants", []):
            allowed_names.add(str(r.get("name")))

        for a in result.get("attractions", []):
            allowed_names.add(str(a.get("name")))

        for f in result.get("flights", {}).get("flights", []):
            allowed_names.add(str(f.get("flight_number")))

        hallucinated = []

        for line in answer.split("\n"):
            if any(char.isdigit() for char in line):  # simple check
                found = any(name in line for name in allowed_names)
                if not found:
                    hallucinated.append(line) 
        print("HALLUCINATED:", hallucinated)
        ## validation code ends here 
        debug = {
            "intent": result.get("intent"),
            "origin": result.get("origin"),
            "destination": result.get("destination"),
            "city": result.get("city"),
            "cuisine": result.get("cuisine"),
            "dates": result.get("dates"),
            "budget_split": result.get("budget_split"),
            "flights_count": len(result.get("flights", {}).get("flights", []))
                if isinstance(result.get("flights"), dict) else 0,
            "hotels_count": len(result.get("hotels", [])),
            "restaurants_count": len(result.get("restaurants", [])),
            "attractions_count": len(result.get("attractions", [])),
            "rag_results_count": len(result.get("rag_results", [])),
            "hallucinated_lines": hallucinated
        }

        return ChatResponse(
            answer=result["answer"],
            debug=debug
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# RUN SERVER
# =========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("travel_langgraph_server:app", host="127.0.0.1", port=8000, reload=True)