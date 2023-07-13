# flake8: noqa
from langchain.prompts import PromptTemplate

template = """The asker is a player of Rust game and is inquiring about GM commands in Rust that can be used to complete quests.
Based on the information from the "Response from window," find a solution to answer the question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.

QUESTION: How can i find my position?
=========
Response from window 24 - 'output_text': 'The command to query your coordinates in Rust is "printpos <Steam64ID>". However, getting another player's position requires admin privileges. \nSOURCES: https://wiki.facepunch.com/rust/useful_commands'
Source: 28-pl
Response from window 4 - 'output_text': 'The document does not mention how to query your coordinates in Rust.\nSOURCES: N/A'
Source: 30-pl
Response from window 8 - 'output_text': 'The command to query your coordinates is not mentioned in the document.\nSOURCES: https://wiki.facepunch.com/rust/useful_commands'
Source: 4-pl
=========
FINAL ANSWER: The command to query your coordinates in Rust is "printpos <Steam64ID>". However, getting another player's position requires admin privileges.
SOURCES: 28-pl

QUESTION: What did the president say about Michael Jackson?
=========
Response from window 4 - 'output_text': 'not mentioned in the document.'
Source: 0-pl
Response from window 18 - 'output_text': 'i don't know.'
Source: 24-pl
Response from window 22 - 'output_text': 'no information from document.'
Source: 5-pl
=========
FINAL ANSWER: i don't know.
SOURCES:

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
