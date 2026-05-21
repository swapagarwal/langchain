from langchain_openai import ChatOpenAI
from typing import Annotated, Optional, Union
from typing_extensions import TypedDict

llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed", max_tokens=200)

class StoryState(TypedDict):
    """State schema for our story generation workflow.
    
    Each field will be populated by parallel nodes:
    - topic: Input from user
    - characters: Generated character descriptions
    - settings: Generated setting description
    - premises: Generated plot premise
    - story_intro: Final combined story introduction
    """
    topic: str
    characters: str
    settings: str
    premises: str
    story_intro: str

from langgraph.graph import StateGraph, START, END
from langchain_core.runnables.config import RunnableConfig

def generate_characters(state: StoryState) -> dict:
    """Generate character descriptions based on topic.
    
    This node runs in parallel with setting and premise generation.
    """
    print("\n Generating characters...")
    prompt = f"Create two character names with brief traits for a story about {state['topic']}"
    msg = llm.invoke(prompt)
    return {"characters": msg.content}

def generate_setting(state: StoryState) -> dict:
    """Generate story setting based on topic.
    
    This node runs in parallel with character and premise generation.
    """
    print(" Generating setting...")
    prompt = f"Describe a vivid setting for a story about {state['topic']}"
    msg = llm.invoke(prompt)
    return {"settings": msg.content}

def generate_premise(state: StoryState) -> dict:
    """Generate story premise based on topic.
    
    This node runs in parallel with character and setting generation.
    """
    print(" Generating premise...")
    prompt = f"Write a one-sentence plot premise for a story about {state['topic']}"
    msg = llm.invoke(prompt)
    return {"premises": msg.content}

def combine_elements(state: StoryState) -> dict:
    """Combine all elements into a cohesive story introduction.
    
    This node only executes after all parallel nodes have completed.
    It receives the full state with all fields populated.
    """
    print("\n Combining elements into final story...")
    
    prompt = f"""
Write a compelling story introduction using these elements:

Characters: {state['characters']}
Setting: {state['settings']}
Premise: {state['premises']}

Weave these together into a captivating opening paragraph.
    """
    
    msg = llm.invoke(prompt)
    return {"story_intro": msg.content}

# Initialize graph with state schema
graph = StateGraph(StoryState)

# Add all nodes
graph.add_node("character", generate_characters)
graph.add_node("setting", generate_setting)
graph.add_node("premise", generate_premise)
graph.add_node("combine", combine_elements)

# Define parallel execution from START
graph.add_edge(START, "character")
graph.add_edge(START, "setting")
graph.add_edge(START, "premise")

# All parallel nodes feed into combine
graph.add_edge("character", "combine")
graph.add_edge("setting", "combine")
graph.add_edge("premise", "combine")

# Combine leads to END
graph.add_edge("combine", END)

# Compile the graph
compiled_graph = graph.compile()
print("\n✅ Graph compiled successfully!")

# png_bytes = compiled_graph.get_graph().draw_mermaid_png()
# with open("graph.png", "wb") as f:
#     f.write(png_bytes)

# # Define initial state
# initial_state = {
#     "topic": "artificial intelligence gaining consciousness"
# }

# print("🚀 Starting parallel story generation...")
# print(f"Topic: {initial_state['topic']}")
# print("="*60)

# # Execute the graph
# result = compiled_graph.invoke(initial_state)

# # Display results
# print("\n" + "="*60)
# print("INTERMEDIATE RESULTS:")
# print("="*60)
# print(f"\nCharacters:\n{result['characters']}")
# print(f"\nSetting:\n{result['settings']}")
# print(f"\nPremise:\n{result['premises']}")

# print("\n" + "="*60)
# print(" FINAL STORY INTRODUCTION:")
# print("="*60)
# print(f"\n{result['story_intro']}")

# Example: Parallel processing with conditional routing

def route_based_on_topic(state: StoryState) -> list[str]:
    """Conditionally determine which parallel nodes to execute."""
    topic = state['topic'].lower()
    
    # Always generate characters
    nodes = ["character"]
    
    # Add setting for sci-fi or fantasy
    if "sci-fi" in topic or "fantasy" in topic:
        nodes.append("setting")
    
    # Add premise for complex stories
    if len(topic.split()) > 3:
        nodes.append("premise")
    
    return nodes
