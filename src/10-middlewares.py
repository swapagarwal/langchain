from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import Literal, Optional
import time

llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

# Sample tools for demonstrations
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for '{query}': Found 3 relevant articles about {query}."

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {to} with subject '{subject}'."

@tool
def delete_file(filename: str) -> str:
    """Delete a file from the system."""
    return f"File '{filename}' deleted successfully."

@tool
def get_user_data(user_id: str) -> str:
    """Retrieve user data from database."""
    # Simulated user data with PII
    return f"User {user_id}: John Doe, email: john.doe@example.com, url: https://www.asdasdas.com, age: 24, city: New York"

print("Tools and model initialized")

from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import MemorySaver

# Create checkpointer for state persistence (required for HITL)
memory = MemorySaver()

# Configure which tools require human approval
hitl_middleware = HumanInTheLoopMiddleware(
    interrupt_on={
        # Require approval for sending emails
        "send_email": {
            "allowed_decisions": ["approve", "edit", "reject"] # True also denotes same, False denotes Safe operation, no approval needed
        },
        # Require approval for file deletion (approve/reject only, no editing)
        "delete_file": {
            "allowed_decisions": ["approve", "reject"]
        }
    }
)

# Create agent with HITL middleware
agent_with_hitl = create_agent(
    llm,
    tools=[search_web, send_email, delete_file],
    middleware=[hitl_middleware],
    checkpointer=memory,  # Required for interrupt/resume functionality
)

print("Human-in-the-loop agent configured")
print("  - send_email: requires approval (can approve, edit, or reject)")
print("  - delete_file: requires approval (can approve or reject)")
print("  - search_web: no approval needed")

# Demonstrating the interrupt/resume flow
from langgraph.types import Command

thread_id = "hitl_demo_thread_1"
config = {"configurable": {"thread_id": thread_id}}

# Step 1: Invoke agent with a request that triggers HITL
print("Step 1: Agent attempts to send email...")
result = agent_with_hitl.invoke(
    {"messages": [{"role": "user", "content": "Send an email to boss@company.com about the project update"}]},
    config=config
)
print(result['__interrupt__'])
# Check if agent is interrupted (waiting for approval)
# The agent will pause before executing send_email
print(f"\nAgent state: {'Interrupted - awaiting approval' if '__interrupt__' in result else 'Completed'}")

# Step 2: Resume with approval
print("\nStep 2: Human approves the action...")
result = agent_with_hitl.invoke(
    Command(
        resume={"decisions": [{"type": "approve"}]}  # or "edit", "reject"
    ),  # Options: "approve", "edit", "reject"
    config=config
)

print("\nFinal result:")
print(result["messages"][-1].content if result.get("messages") else "Action completed")
for message in result['messages']:
    message.pretty_print()
print("\n")

# from langchain.agents.middleware import PIIMiddleware

# # Redact emails in both inputs and outputs
# pii_email = PIIMiddleware(
#     "email",
#     strategy="redact",
#     apply_to_input=True,
#     apply_to_output=True,
# )

# # Redact SSNs - critical for compliance
# pii_ssn = PIIMiddleware(
#     "credit_card",
#     strategy="redact",
#     apply_to_output=True,  # Mainly concerned about leaking in outputs
# )

# # Redact phone numbers
# pii_phone = PIIMiddleware(
#     "url",
#     strategy="redact",
#     apply_to_output=True,
# )

# # Create agent with multiple PII middleware (they stack)
# agent_with_pii = create_agent(
#     llm,
#     tools=[get_user_data, search_web],
#     middleware=[pii_email, pii_ssn, pii_phone],
# )

# print("PII-protected agent configured")
# print("  - Emails: redacted in input and output")
# print("  - url: redacted in output")
# print("  - credit_card: redacted in output")

# # Test PII redaction - the tool returns sensitive data, middleware redacts it
# result = agent_with_pii.invoke({
#     "messages": [{"role": "user", "content": "Get user data for user ID 12345"}]
# })

# print("Agent response (PII should be redacted):")
# for message in result['messages']:
#     message.pretty_print()


# Note: The get_user_data tool returns:
# "User 12345: John Doe, email: john.doe@example.com, SSN: 123-45-6789, phone: 555-123-4567"
# After PII middleware, sensitive data will be replaced with redaction placeholders
