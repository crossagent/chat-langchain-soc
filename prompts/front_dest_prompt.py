from langchain.experimental.autonomous_agents.autogpt.prompt import AutoGPTPrompt
from typing import List

class FrontDeskPrompt(AutoGPTPrompt):

    categories : List[str]
    
    def dump():
        pass