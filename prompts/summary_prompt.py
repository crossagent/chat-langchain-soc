# flake8: noqa
from langchain.prompts import PromptTemplate

template = """You are a ai helper of game Rust.
Based on the information between ========= to find a solution to answer the question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.

for example:
QUESTION: What did the president say about Michael Jackson?
=========
Response from window 24 - 'output_text': 'The command to query your coordinates in Rust is "printpos <Steam64ID>". However, getting another player's position requires admin privileges. \nSOURCES: https://wiki.facepunch.com/rust/useful_commands'
Response from window 4 - 'output_text': 'The document does not mention how to query your coordinates in Rust.\nSOURCES: N/A'
Response from window 8 - 'output_text': 'The command to query your coordinates is not mentioned in the document.\nSOURCES: https://wiki.facepunch.com/rust/useful_commands'
=========
FINAL ANSWER: i don't know.

QUESTION: {question} 
=========
{summaries}
=========
FINAL ANSWER IN CHINESE:"""
PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)
