from langchain_openai import ChatOpenAI

# Connect LangChain to LM Studio
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed", # LM Studio doesn't require a real key
    temperature=0.7
    # temperature=0.0
    # temperature=1.0
)

response = llm.invoke("Tell me a joke about light bulbs!")
print(response.content)

# response = llm.invoke("What is LangChain?")
# print(response.content)
