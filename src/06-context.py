from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import json

llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed", max_tokens=200)

# First chain generates a story
story_prompt = ChatPromptTemplate.from_template("Write a short story about {topic}")
story_chain = (story_prompt | llm | StrOutputParser()).with_config(tags=["story"])

# Second chain analyzes the story
analysis_prompt = ChatPromptTemplate.from_template("Analyze the following story's mood:\n{story}")
analysis_chain = (analysis_prompt | llm | StrOutputParser()).with_config(tags=["analysis"])

# Using RunnablePassthrough.assign to preserve data
enhanced_chain = RunnablePassthrough.assign(
    story=story_chain # Add 'story' key with generated content
    ).assign(
    analysis=analysis_chain # Add 'analysis' key with analysis of the story
)

# Execute the chain
result = enhanced_chain.invoke({"topic": "a rainy day"})
print(result.keys())
print(json.dumps(
    result,
    indent=4,
    separators=(',', ': ')
))
