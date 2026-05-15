from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

chat = ChatOpenAI(base_url="http://localhost:1234/v1",
    api_key="not-needed",
    temperature=0.7,
    # max_tokens=200, # limit output length for all calls
    # timeout=30,     # Request timeout
    # max_retries=2,  # Retry failed requests
)

messages = [
    SystemMessage(content="You're a helpful programming assistant"), # Sets behavior and context for the model
    HumanMessage(content="Write a Python function to calculate factorial"), # Represents user input like questions, commands, and data
    # AIMessage: Contains model responses
    # ToolMessage: Results from tool/function calls (covered later)
]

response = chat.stream(messages)
# response = chat.bind(max_tokens=200).stream(messages) # limit output length just for this call
for chunk in response:
    print(chunk.content, end="", flush=True)
print()

"""
LangChain uses message objects instead of raw strings to represent conversational interactions.

Why Messages?
Structure: Clear separation of roles (system, user, assistant)
Metadata: Attach additional info (timestamps, sources, tool calls)
Type Safety: Easier to validate and debug
"""

# # Messages can carry metadata
# message_with_metadata = HumanMessage(
#     content="Translate this to French: Hello, world!",
#     additional_kwargs={"user_id": "12345", "session": "abc"}
# )

# response = chat.invoke([message_with_metadata])

# # Inspect response metadata
# print("Content:", response.content)
# print("\nResponse Metadata:")
# print(f"  Model: {response.response_metadata.get('model_name')}")
# print(f"  Tokens used: {response.response_metadata.get('token_usage')}")
# print(f"  Finish reason: {response.response_metadata.get('finish_reason')}")
