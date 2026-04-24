from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3:8b")
print(llm.invoke("Say hello in one sentence."))