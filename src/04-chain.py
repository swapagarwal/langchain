from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

chat = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

template = ChatPromptTemplate.from_messages([
    ("system", "You're a helpful programming assistant"),
    ("user", "{problem}")
])

chain = template | chat # Create and run a chain

response = chain.stream({"problem": "Write a Python function to calculate factorial"})
for chunk in response:
    print(chunk.content, end="", flush=True)
print()
