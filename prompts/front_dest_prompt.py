from langchain.experimental.autonomous_agents.autogpt.prompt import AutoGPTPrompt
from typing import List, Any
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

class FrontDeskPrompt(AutoGPTPrompt):
    
    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        return super().format_messages(**kwargs)