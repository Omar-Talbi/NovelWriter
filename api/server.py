from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from pathlib import Path
from pydantic import BaseModel
import uvicorn
from generation.drafter import generate_draft
from rag.story_bible import load_story_bible, Fact
import time

app = FastAPI()
BIBLE_PATH = Path('rag/story_bible.json')


class Outline(BaseModel):
    outline: str


@app.post('/draft')
async def draft(outline: Outline):
    text = generate_draft(outline.outline, BIBLE_PATH)
    def iter_text():
        for word in text.split():
            yield word + ' '
    return StreamingResponse(iter_text(), media_type='text/plain')


class FactInput(BaseModel):
    id: str
    type: str
    text: str


@app.post('/facts')
async def add_fact(fact: FactInput):
    bible = load_story_bible(BIBLE_PATH)
    bible.add_fact(Fact(id=fact.id, type=fact.type, text=fact.text, timestamp=time.time()))
    return JSONResponse({'status': 'ok'})


@app.get('/health')
async def health():
    return {'status': 'ok'}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
