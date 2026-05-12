from langchain_openai import ChatOpenAI

llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

response = llm.stream("What is LangChain?")
for chunk in response:
    print(chunk.content, end="", flush=True)
print()
