"""Main entrypoint for the app."""
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore

from callbacks.callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from callbacks.socWebCallBacks import StreamSocLLMCallbackHandler
from query_data import get_chain
from schemas import ChatResponse

from agents.rustserver_cmd_agent import get_rust_server_cmd_gpt

app = FastAPI()
templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None


@app.on_event("startup")
async def startup_event():
    logging.info("loading vectorstore")
    #if not Path("vectorstore.pkl").exists():
        #raise ValueError("vectorstore.pkl does not exist, please run ingest.py first")
    #with open("vectorstore.pkl", "rb") as f:
        #global vectorstore
        #vectorstore = pickle.load(f)


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    qa_chain = get_chain(vectorstore, question_handler, stream_handler)
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)

    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            result = await qa_chain.acall(
                {"question": question, "chat_history": chat_history}
            )
            chat_history.append((question, result['text']))

            print("answer is" + result['text'])

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())

@app.websocket("/chatSoc")
async def websocket_endpoint_soc(websocket: WebSocket):
    await websocket.accept()

    print("run chatsoc")

    stream_handler = StreamSocLLMCallbackHandler(websocket)

    #soc_agent from chat.py
    config = {"stream_handler" : stream_handler, "websocket" : websocket}
    soc_agent = get_rust_server_cmd_gpt(**config)

    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())
            
            result = await soc_agent.arun(question.strip())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            # 发送最终的询问结果
            result_resp = ChatResponse(sender="bot", message="运行结束:" + result, type="stream")
            await websocket.send_json(result_resp.dict())

            # 结束对话
            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict()) 
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
