from fastapi import FastAPI,HTTPException,WebSocket,Request
from fastapi.responses import HTMLResponse
from models.models import Queryhandler
from generate import *
from scrapper import scrapper
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
def startup():
    init_models() 


@app.post("/query/")
def search_query(data:Queryhandler):
    if not data.query:
        raise HTTPException(status_code=400,data="Invalid query")
    result = answer(data.query.strip(),conversation_memory)
    return {"status":"OK","query":data,"answer":result}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        print("receied")
        user_message = await websocket.receive_text()
        # yaha tumhara LLaMA / RAG / LoRA logic lagega
        result = answer(user_message.strip(),[])
        await websocket.send_text(result)

@app.get("/")
def index(request:Request,response_class=HTMLResponse):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )
