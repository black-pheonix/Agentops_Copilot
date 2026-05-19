from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
import asyncio
import json
import os
import sys

app = FastAPI()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import run_agent

class Question(BaseModel):
    text:str

    @field_validator("text")
    @classmethod
    def text_validation(cls, value):
        if not value.strip():
            raise ValueError("Question cannot be empty")
        return value.strip()

class Answer(BaseModel):
    question:str
    answer:str
@app.get("/health")
def health():
    return {"status": "ok", "service": "AgentOps Copilot"}

@app.post("/ask", response_model=Answer)
def ask(question: Question):
    try:
        answer = run_agent(question.text)
        return Answer(question=question.text, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ask/stream")
async def ask_stream(text: str):
    if not text.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty")

    async def event_generator():
        yield f"data: {json.dumps({'event': 'status', 'message': 'Searching documentation...'})}\n\n"
        
        import concurrent.futures
        loop = asyncio.get_event_loop()
        
        with concurrent.futures.ThreadPoolExecutor() as pool:
            answer = await loop.run_in_executor(pool, run_agent, text)
        
        yield f"data: {json.dumps({'event': 'answer', 'answer': answer})}\n\n"
        yield f"data: {json.dumps({'event': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )