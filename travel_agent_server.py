import os
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
import snowflake.connector
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama

load_dotenv()


# =========================
# Logging
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("travel-agent")
# =========================
# Config
# =========================

SNOWFLAKE_CONFIG = {
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE", "TRAVEL_RAG"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
}

FLIGHTS_FILE = os.getenv("FLIGHTS_FILE", "flights.csv")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# Historical coverage from your dataset
SUPPORTED_FLIGHT_YEAR = 2022
SUPPORTED_FLIGHT_MONTHS = {2, 4, 7}

# Inclusive day count:
# 2022-04-10 to 2022-04-12 => 3 days
INCLUSIVE_DAY_COUNT = True

# =========================
# FastAPI
# =========================

app = FastAPI(title="Travel AI Agent", version="1.0")

# =========================
# Request / Response
# =========================

class AgentRequest(BaseModel):
    message: str = Field(..., description="Natural language user request")


class AgentResponse(BaseModel):
    answer: str

# =========================
# Utility functions
# =========================

def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def calculate_days(start_date: str, end_date: str) -> int:
    start = parse_date(start_date)
    end = parse_date(end_date)
    delta = (end - start).days
    if INCLUSIVE_DAY_COUNT:
        delta += 1
    if delta <= 0:
        raise ValueError("end_date must be after or equal to start_date")
    return delta


def list_trip_dates(start_date: str, end_date: str) -> list[str]:
    start = parse_date(start_date)
    end = parse_date(end_date)
    if end < start:
        raise ValueError("end_date must be after or equal to start_date")

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


def is_supported_flight_date(date_str: str) -> bool:
    dt = parse_date(date_str)
    return dt.year == SUPPORTED_FLIGHT_YEAR and dt.month in SUPPORTED_FLIGHT_MONTHS


def sql_escape(value: str) -> str:
    return value.replace("'", "''")


def safe_float(x: Any) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None

# =========================
# Snowflake connection
# =========================

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
# Local flights data
# =========================

def load_flights_df() -> pd.DataFrame:
    df = pd.read_csv(FLIGHTS_FILE)

    # Normalize column names
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
        raise ValueError(f"Missing required flight columns: {missing}")

    # Text normalization
    df["Flight Number"] = df["Flight Number"].astype(str).str.strip()
    df["OriginCityName"] = df["OriginCityName"].astype(str).str.strip()
    df["DestCityName"] = df["DestCityName"].astype(str).str.strip()
    df["DepTime"] = df["DepTime"].astype(str).str.strip()
    df["ArrTime"] = df["ArrTime"].astype(str).str.strip()
    df["ActualElapsedTime"] = df["ActualElapsedTime"].astype(str).str.strip()

    # Numeric cleanup
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")

    # Date cleanup: your sample uses DD-MM-YYYY
    df["FlightDate"] = pd.to_datetime(df["FlightDate"], format="%d-%m-%Y", errors="coerce")

    # Keep time as string for now because your source is time-only.
    # You can convert to time type later if needed.
    return df


FLIGHTS_DF = load_flights_df()
# =========================
# Tools
# =========================

@tool
def compute_trip_constraints(start_date: str, end_date: str, budget: float) -> str:
    """
    Compute trip duration and budget split from start_date, end_date, and total budget.
    Use this first for itinerary requests with dates and budget.
    Returns days, exact trip dates, and budget allocation.
    """
    days = calculate_days(start_date, end_date)
    dates = list_trip_dates(start_date, end_date)

    # Simple default allocation. You can tune later.
    hotel_budget_total = budget * 0.45
    food_budget_total = budget * 0.25
    activity_budget_total = budget * 0.15
    flight_budget_total = budget * 0.15

    result = {
        "days": days,
        "dates": dates,
        "budget": budget,
        "allocation": {
            "hotel_budget_total": round(hotel_budget_total, 2),
            "hotel_budget_per_night": round(hotel_budget_total / max(days, 1), 2),
            "food_budget_total": round(food_budget_total, 2),
            "food_budget_per_day": round(food_budget_total / max(days, 1), 2),
            "activity_budget_total": round(activity_budget_total, 2),
            "flight_budget_total": round(flight_budget_total, 2),
        },
    }
    return json.dumps(result, ensure_ascii=False, indent=2)

@tool
def search_flights(origin: str, destination: str, start_date: str, max_price: Optional[float] = None, limit: int = 5) -> str:
    """
    Search local flights CSV for flights matching origin, destination, and start_date.
    The dataset only supports February, April, and July 2022.
    """
    if not is_supported_flight_date(start_date):
        return json.dumps(
            {
                "status": "unsupported_date",
                "message": "Flight data is available only for February, April, and July 2022."
            },
            ensure_ascii=False,
            indent=2,
        )

    travel_day = parse_date(start_date)

    df = FLIGHTS_DF.copy()
    df = df[
        df["OriginCityName"].str.contains(origin, case=False, na=False)
        & df["DestCityName"].str.contains(destination, case=False, na=False)
    ]
    df = df[df["FlightDate"] == travel_day]

    if max_price is not None:
        df = df[df["Price"].notna() & (df["Price"] <= max_price)]

    df = df.sort_values(["Price", "Distance"], ascending=[True, True]).head(limit)

    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "flight_number": row["Flight Number"],
                "price": safe_float(row["Price"]),
                "departure_time": row["DepTime"],
                "arrival_time": row["ArrTime"],
                "duration": row["ActualElapsedTime"],
                "flight_date": row["FlightDate"].strftime("%Y-%m-%d") if pd.notna(row["FlightDate"]) else None,
                "origin_city": row["OriginCityName"],
                "destination_city": row["DestCityName"],
                "distance": safe_float(row["Distance"]),
            }
        )

    return json.dumps(records, ensure_ascii=False, indent=2)

@tool
def search_hotels(city: str, max_price_per_night: Optional[float] = None, min_rating: float = 0.0, limit: int = 5) -> str:
    """
    Search hotels from Snowflake by city, optional nightly budget, and minimum rating.
    """
    city_esc = sql_escape(city)

    where_price = ""
    if max_price_per_night is not None:
        where_price = f" AND price IS NOT NULL AND price <= {max_price_per_night}"

    query = f"""
        SELECT
            name, city, price, rating, room_type, max_occupancy, min_nights, text
        FROM travel_embeddings
        WHERE type = 'hotel'
          AND city ILIKE '%{city_esc}%'
          AND rating >= {min_rating}
          {where_price}
        ORDER BY price ASC NULLS LAST, rating DESC
        LIMIT {limit}
    """

    rows = fetch_rows(query)
    payload = [
        {
            "name": row[0],
            "city": row[1],
            "price_per_night": row[2],
            "rating": row[3],
            "room_type": row[4],
            "max_occupancy": row[5],
            "min_nights": row[6],
            "text": row[7],
        }
        for row in rows
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)

@tool
def search_restaurants(city: str, max_average_cost: Optional[float] = None, min_rating: float = 0.0, limit: int = 8) -> str:
    """
    Search restaurants from Snowflake by city, optional average cost cap, and minimum rating.
    """
    city_esc = sql_escape(city)

    where_cost = ""
    if max_average_cost is not None:
        where_cost = f" AND average_cost IS NOT NULL AND average_cost <= {max_average_cost}"

    query = f"""
        SELECT
            name, city, average_cost, rating, cuisines, text
        FROM travel_embeddings
        WHERE type = 'restaurant'
          AND city ILIKE '%{city_esc}%'
          AND rating >= {min_rating}
          {where_cost}
        ORDER BY average_cost ASC NULLS LAST, rating DESC
        LIMIT {limit}
    """

    rows = fetch_rows(query)
    payload = [
        {
            "name": row[0],
            "city": row[1],
            "average_cost": row[2],
            "rating": row[3],
            "cuisines": row[4],
            "text": row[5],
        }
        for row in rows
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)
@tool
def search_attractions(city: str, preference: str = "best attractions", limit: int = 8) -> str:
    """
    Search attractions from Snowflake using semantic retrieval plus city filtering.
    Use this for sightseeing, things to do, family places, romantic places, and similar queries.
    """
    city_esc = sql_escape(city)
    pref_esc = sql_escape(f"{preference} in {city}")

    query = f"""
        SELECT
            name, city, address, phone_number, website, rating, text
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
    payload = [
        {
            "name": row[0],
            "city": row[1],
            "address": row[2],
            "phone_number": row[3],
            "website": row[4],
            "rating": row[5],
            "text": row[6],
        }
        for row in rows
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)
@tool
def semantic_travel_search(query_text: str, city: Optional[str] = None, entity_type: Optional[str] = None, limit: int = 5) -> str:
    """
    General semantic travel search over Snowflake.
    Use this for specific natural-language questions that don't fit the other tools directly.
    """
    filters = []

    if city:
        city_esc = sql_escape(city)
        filters.append(f"city ILIKE '%{city_esc}%'")

    if entity_type:
        et_esc = sql_escape(entity_type)
        filters.append(f"type = '{et_esc}'")

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
    payload = [
        {
            "name": row[0],
            "city": row[1],
            "type": row[2],
            "text": row[3],
        }
        for row in rows
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)

# =========================
# Agent model
# =========================

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0,
    num_ctx=8192,
    # request_timeout is supported by ChatOllama constructor params pattern
)

SYSTEM_PROMPT = """
You are a production travel planning AI agent.

You must understand user intent and choose tools dynamically.

Behavior rules:
1. For itinerary planning with start_date, end_date, and budget:
   - call compute_trip_constraints first
   - if origin is provided, call search_flights
   - call search_hotels
   - call search_restaurants
   - call search_attractions
   - produce a day-by-day itinerary with exact dates
2. If a user asks a specific question, only call the minimum tools needed.
3. Never invent flights, hotels, restaurants, or attractions.
4. Hotel price means price per night.
5. Restaurant average_cost means dining cost.
6. Flights are historical local data, not live booking data.
7. If flight date is unsupported, clearly say so and continue the rest of the plan if possible.
8. Use budgets realistically. Prefer cheaper options when the user asks for budget or cheap options.
9. Keep answers concise but useful.

For itinerary output, use this structure:
Flight Details:
...
Recommended Hotel:
...
Day 1 (YYYY-MM-DD):
- Morning:
- Lunch:
- Afternoon:
- Evening:
...
Total Cost Summary:
...
"""

agent = create_agent(
    model=llm,
    tools=[
        compute_trip_constraints,
        search_flights,
        search_hotels,
        search_restaurants,
        search_attractions,
        semantic_travel_search,
    ],
    system_prompt=SYSTEM_PROMPT,
)

# =========================
# API
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=AgentResponse)
def chat(req: AgentRequest):
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": req.message}]}
        )

        messages = result.get("messages", [])
        if not messages:
            raise HTTPException(status_code=500, detail="Agent returned no messages")

        final_message = messages[-1]
        content = getattr(final_message, "content", None)

        if isinstance(content, list):
            answer = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
            ).strip()
        else:
            answer = str(content).strip()

        return AgentResponse(answer=answer)

    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="flights.csv not found")
    except snowflake.connector.errors.Error as e:
        raise HTTPException(status_code=500, detail=f"Snowflake error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("travel_agent_server:app", host="127.0.0.1", port=8000, reload=True)