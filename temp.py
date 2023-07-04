import re

llm_output = """`Thought: I need to clarify the user's problem and find out the tools, task, and type of help they need.
Action: HumanInput
Action Input in Chinese:请问您遇到的问题是什么？`"""

# Parse out the action and action input
#regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*in\s*chinese\s*:[\s]*(.*)"
regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*in\s*chinese\s*:(.*)"
match = re.search(regex, llm_output, re.DOTALL)

#if not match:
    #raise Exception(f"Could not parse LLM output: `{llm_output}`")