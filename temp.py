import re
import json

#if not match:
    #raise Exception(f"Could not parse LLM output: `{llm_output}`")

test = """{\'output_text\': "The specific method or command to find a character\'s location in Rust is not mentioned in the provided extracted parts of the document. It is recommended to refer to the Rust Admin Commands List for the appropriate command. \\nSOURCES: \\n- https://www.corrosionhour.com/rust-admin-commands/"}"""

data = eval(test)

json_string = test.replace("\'", "'").replace('\\"', '"').replace("\\n", "\n")

data = json.loads(json_string)

print(data["output_text"])
