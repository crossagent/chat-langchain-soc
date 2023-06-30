from langchain.callbacks.base import AsyncCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage


class MyCustomHandler(AsyncCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"My custom handler, token: {token}")


# To enable streaming, we pass in `streaming=True` to the ChatModel constructor
# Additionally, we pass in a list with our custom handler
chat = ChatOpenAI(max_tokens=25, streaming=True, callbacks=[MyCustomHandler()])

chat([HumanMessage(content="Write me a song about sparkling water.")])