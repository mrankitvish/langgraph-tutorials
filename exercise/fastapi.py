from fastapi import FastAPI
from pydantic import BaseModel, Field

class myapp(BaseModel):
    name: str
    age: int = Field(..., gt=0)

app = FastAPI()

@app.post("/test/")
async def test(myapp: myapp):
    return myapp