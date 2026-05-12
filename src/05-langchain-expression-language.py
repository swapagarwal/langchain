from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Create components
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed", max_tokens=200)
output_parser = StrOutputParser()

# Chain them together using LCEL
chain = prompt | llm | output_parser # Basic sequential chain: Just prompt to LLM

# Execute the workflow with a single call
result = chain.invoke({"topic": "programming"})
print(result)

"""
with_transformation = prompt | llm | (lambda x: x.upper()) | StrOutputParser()
decision_chain = prompt | llm | (lambda x: route_based_on_content(x)) | {
    "summarize": summarize_chain,
    "analyze": analyze_chain
}
# """

# # First chain generates a story
# story_prompt = ChatPromptTemplate.from_template("Write a short story about {topic}")
# story_chain = (story_prompt | llm | StrOutputParser()).with_config(tags=["story"])

# # Second chain analyzes the story
# analysis_prompt = ChatPromptTemplate.from_template("Analyze the following story's mood:\n{story}")
# analysis_chain = (analysis_prompt | llm | StrOutputParser()).with_config(tags=["analysis"])

# # Combine chains
# story_with_analysis = story_chain | analysis_chain

# # Run the combined chain
# story_analysis = story_with_analysis.invoke({"topic": "a rainy day"})
# print("\nAnalysis:", story_analysis)
