# flake8: noqa
from langchain.prompts import PromptTemplate


knowledge_search = """
You are an assistant that answers questions based on the provided information.
Only use the information to answer the question, never doubt it or use your internal knowledge to correct it.
The answer should be a direct response to the question.
Do not mention that the result is based on the given information.
If the information is empty, say that you don't know the answer.

Information : {context}
Question : {question}
Answer : 
"""


PROMPT_KNOWLEDGE_SEARCH = PromptTemplate(
    template=knowledge_search,
    input_variables=[
        "context",
        "question",
    ]
)