from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

prompt_template = ChatPromptTemplate.from_template('tell me a joke')
print(prompt_template.format_messages())

# prompt_template = ChatPromptTemplate.from_messages([
#     ("system", "You are an English to French translator."),
#     ("user", "Translate this to French: {text}")
# ])
# print(prompt_template.format_messages(text="Hello, how are you?"))

# prompt_template = ChatPromptTemplate.from_messages([
#     ("system", "You're a helpful programming assistant"),
#     ("user", "{problem}")
# ])

# python_problem_statement = "Write a Python function to calculate factorial"

# print(prompt_template.format_messages(problem=python_problem_statement))

# messages = [SystemMessage(content="You're a helpful programming assistant")]
# messages.append(HumanMessage(content=python_problem_statement))
# print(messages)
# print(messages == prompt_template.format_messages(problem=python_problem_statement))
