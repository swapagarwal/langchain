# Core LangChain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Agent creation
from langchain.agents import create_agent

# Memory and state management
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# For structured output
from pydantic import BaseModel, Field
from typing import List, Optional, Annotated, Literal

# Utilities
import json
from datetime import datetime

llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

@tool
def calculator(
    operation: Annotated[Literal["add", "subtract", "multiply", "divide"], "The math operation"],
    a: Annotated[float, "First number"],
    b: Annotated[float, "Second number"]
) -> str:
    """Perform basic arithmetic operations. Use for exact calculations."""
    ops = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else "Error: Division by zero"
    }
    result = ops.get(operation.lower(), "Invalid operation")
    return f"Result: {result}"

@tool
def get_current_time(timezone: str = "local") -> str:
    """Get the current date and time."""
    current = datetime.now()
    return f"Current time: {current.strftime('%Y-%m-%d %H:%M:%S')}"

@tool
def search_database(query: Literal["laptop", "phone", "tablet"]) -> str:
    """Search a simulated product database. Use when user asks about products or inventory."""
    # Simulated product database
    products = {
        "laptop": "In stock: 15 units, Price: $999",
        "phone": "In stock: 42 units, Price: $699",
        "tablet": "Out of stock, Expected: Next week",
    }
    result = products.get(query.lower(), f"No product found matching '{query}'")
    return result

tools = [calculator, get_current_time, search_database]

print(f"Model and {len(tools)} tools ready")

system_prompt = """You are a helpful assistant with access to tools.

When a user asks a question:
1. Think about which tool(s) you need
2. Call the appropriate tools
3. Provide a clear, concise answer

Be direct and avoid unnecessary explanations of your process."""

# Create the agent
# In LangChain v1, create_agent is the standard way to build agents
agent = create_agent(
    llm,
    tools,
    system_prompt=system_prompt,
)

print("Agent created")
print(f"Agent type: {type(agent)}")

# Single query - agent has no memory of previous interactions
query = "What is current date and time?"

response = agent.invoke({"messages": [HumanMessage(content=query)]})

print(f"Query: {query}")
print(f"\nAgent response:")
print(response["messages"][-1].content)

[msg.pretty_print() for msg in response["messages"]]

# Query that requires multiple tools
query = "What is the laptop price in the database? After getting the price, calculate 15% of the price."

response = agent.invoke({"messages": [HumanMessage(content=query)]})

print(f"Query: {query}")
print(f"\nAgent response:")
print(response["messages"][-1].content)
[msg.pretty_print() for msg in response["messages"]]
