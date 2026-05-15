from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Create components
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed", max_tokens=200)
output_parser = StrOutputParser()

# Chain them together using LCEL
chain = prompt | llm | output_parser # Basic sequential chain: Just prompt to LLM

# print("Chain structure:")
# print(chain)

# # Get input/output schema (useful for debugging)
# print("\nInput schema:")
# print(chain.input_schema.schema())

# print("\nOutput schema:")
# print(chain.output_schema.schema())

# Execute the workflow with a single call
result = chain.invoke({"topic": "programming"})
print(result)

# chain_with_transformation = prompt | llm | StrOutputParser() | (lambda x: x.upper())
# result = chain_with_transformation.invoke({"topic": "programming"})
# print(result)

"""
decision_chain = prompt | llm | (lambda x: route_based_on_content(x)) | {
    "summarize": summarize_chain,
    "analyze": analyze_chain
}
"""

# First chain generates a story
story_prompt = ChatPromptTemplate.from_template("Write a short story about {topic}")
story_chain = (story_prompt | llm | StrOutputParser())

# Second chain analyzes the story
analysis_prompt = ChatPromptTemplate.from_template("Analyze the following story's mood:\n{story}")
analysis_chain = (analysis_prompt | llm | StrOutputParser())

# Combine chains
story_with_analysis = story_chain | analysis_chain
# story_prompt | llm | StrOutputParser() | analysis_prompt | llm | StrOutputParser()

# Run the combined chain
# story_analysis = story_with_analysis.invoke({"topic": "a rainy day"})
# print("\nAnalysis:", story_analysis)
