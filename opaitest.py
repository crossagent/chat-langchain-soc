
from langchain.llms import OpenAI

llm = OpenAI()


result = llm("openaai status test")

print(result)