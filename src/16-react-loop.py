import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from rich.console import Console
from rich.markdown import Markdown

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Agentic Architecture - ReAct"

llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

# LM Studio only accepts string tool_choice (none/auto/required), not OpenAI's
# per-function object form that LangChain uses when forcing a specific tool name.
_LOCAL_STRUCTURED_KWARGS = {"method": "function_calling", "tool_choice": "required"}

# Initialize console for pretty printing
console = Console()

# Define the state for our graphs
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Define the tool and LLM
search_tool = TavilySearchResults(max_results=2, name="web_search")
llm_with_tools = llm.bind_tools([search_tool])

# Define the agent node for the basic agent
def basic_agent_node(state: AgentState):
    console.print("--- BASIC AGENT: Thinking... ---")
    # Note: We provide a system prompt to encourage it to answer directly after one tool call
    system_prompt = "You are a helpful assistant. You have access to a web search tool. Answer the user's question based on the tool's results. You must provide a final answer after one tool call."
    messages = [("system", system_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Define the basic, linear graph
basic_graph_builder = StateGraph(AgentState)
basic_graph_builder.add_node("agent", basic_agent_node)
basic_graph_builder.add_node("tools", ToolNode([search_tool]))

basic_graph_builder.set_entry_point("agent")
# After the agent, it can only go to tools, and after tools, it MUST end.
basic_graph_builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": "__end__"})
basic_graph_builder.add_edge("tools", END)

basic_tool_agent_app = basic_graph_builder.compile()

print("Basic single-shot tool-using agent compiled successfully.")
multi_step_query = "Who is the current CEO of the company that created the sci-fi movie 'Dune', and what was the budget for that company's most recent film?"

console.print(f"[bold yellow]Testing BASIC agent on a multi-step query:[/bold yellow] '{multi_step_query}'\n")

basic_agent_output = basic_tool_agent_app.invoke({"messages": [("user", multi_step_query)]})

console.print("\n--- [bold red]Final Output from Basic Agent[/bold red] ---")
console.print(Markdown(basic_agent_output['messages'][-1].content))

def react_agent_node(state: AgentState):
    console.print("--- REACT AGENT: Thinking... ---")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# The ToolNode is the same as before
react_tool_node = ToolNode([search_tool])

# The router is also the same logic
def react_router(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        console.print("--- ROUTER: Decision is to call a tool. ---")
        return "tools"
    console.print("--- ROUTER: Decision is to finish. ---")
    return "__end__"

# Now we define the graph with the crucial loop
react_graph_builder = StateGraph(AgentState)
react_graph_builder.add_node("agent", react_agent_node)
react_graph_builder.add_node("tools", react_tool_node)

react_graph_builder.set_entry_point("agent")
react_graph_builder.add_conditional_edges("agent", react_router, {"tools": "tools", "__end__": "__end__"})

# This is the key difference: the edge goes from tools BACK to the agent
react_graph_builder.add_edge("tools", "agent")

react_agent_app = react_graph_builder.compile()
print("ReAct agent compiled successfully with a reasoning loop.")

console.print(f"[bold green]Testing ReAct agent on the same multi-step query:[/bold green] '{multi_step_query}'\n")

final_react_output = None
for chunk in react_agent_app.stream({"messages": [("user", multi_step_query)]}, stream_mode="values"):
    final_react_output = chunk
    console.print(f"--- [bold purple]Current State[/bold purple] ---")
    chunk['messages'][-1].pretty_print()
    console.print("\n")

console.print("\n--- [bold green]Final Output from ReAct Agent[/bold green] ---")
console.print(Markdown(final_react_output['messages'][-1].content))

# Visualize the graph
png_bytes = react_agent_app.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_bytes)

class TaskEvaluation(BaseModel):
    """Schema for evaluating an agent's ability to complete a task."""
    task_completion_score: int = Field(description="Score 1-10 on whether the agent successfully completed all parts of the user's request.")
    reasoning_quality_score: int = Field(description="Score 1-10 on the logical flow and reasoning process demonstrated by the agent.")
    justification: str = Field(description="A brief justification for the scores.")

judge_llm = llm.with_structured_output(TaskEvaluation, **_LOCAL_STRUCTURED_KWARGS)

def evaluate_agent_output(query: str, agent_output: dict):
    trace = "\n".join([f"{m.type}: {m.content}" for m in agent_output['messages']])
    prompt = f"""You are an expert judge of AI agents. Evaluate the following agent's performance on the given task on a scale of 1-10. A score of 10 means the task was completed perfectly. A score of 1 means complete failure.

    **User's Task:**
    {query}

    **Full Agent Conversation Trace:**
    ```
    {trace}
    ```
    """
    return judge_llm.invoke(prompt)

console.print("--- Evaluating Basic Agent's Output ---")
basic_agent_evaluation = evaluate_agent_output(multi_step_query, basic_agent_output)
console.print(basic_agent_evaluation.model_dump())

console.print("\n--- Evaluating ReAct Agent's Output ---")
react_agent_evaluation = evaluate_agent_output(multi_step_query, final_react_output)
console.print(react_agent_evaluation.model_dump())

"""
1. Receive Goal
The agent is given a complex task that can’t be solved in one step.
2. Think (Reason)
The agent generates a thought, like: “To answer this, I first need to find piece of information X”.
3. Act
Based on that thought, it executes an action, like calling a search tool for ‘X’.
4. Observe
The agent gets the result for ‘X’ back from the tool.
5. Repeat
It takes that new information and goes back to step 2, thinking: “Okay, now that I have X, I need to use it to find Y”. This loop continues until the final goal is met.
"""
