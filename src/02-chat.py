from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

chat = ChatOpenAI(base_url="http://localhost:1234/v1",
    api_key="not-needed",
    temperature=0.7,
    # max_tokens=200, # limit output length for all calls
)

messages = [
    SystemMessage(content="You're a helpful programming assistant"), # Sets behavior and context for the model
    HumanMessage(content="Write a Python function to calculate factorial"), # Represents user input like questions, commands, and data
    # AIMessage: Contains model responses
]

response = chat.stream(messages)
# response = chat.bind(max_tokens=200).stream(messages) # limit output length just for this call
for chunk in response:
    print(chunk.content, end="", flush=True)
print()
