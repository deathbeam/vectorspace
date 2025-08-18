import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from .logger import setup_logger
from .core import VectorspaceCore


class Watch(BaseModel):
    dir: str


class Query(Watch):
    text: str
    max: int


class QueryData(BaseModel):
    filename: str
    content: str
    score: float
    start_row: int = 0
    end_row: int = 0
    start_col: int = 0
    end_col: int = 0


app = FastAPI(title="vectorspace")
vectorspace_core = VectorspaceCore()


@app.post("/start")
def start(watch: Watch):
    return vectorspace_core.start_watching(watch.dir)


@app.post("/stop")
def stop(watch: Watch):
    vectorspace_core.stop_watching(watch.dir)
    return watch


@app.post("/clear")
def clear(watch: Watch):
    vectorspace_core.clear(watch.dir)
    return watch


@app.post("/count")
def count(watch: Watch) -> int:
    return vectorspace_core.count_files(watch.dir)


@app.post("/query")
def query(query: Query) -> List[QueryData]:
    results = vectorspace_core.query(query.dir, query.text, query.max)

    return [
        QueryData(
            filename=result["filename"],
            content=result["content"],
            score=result["score"],
            start_row=result.get("start_row", 0),
            end_row=result.get("end_row", 0),
            start_col=result.get("start_col", 0),
            end_col=result.get("end_col", 0),
        )
        for result in results
    ]


def main():
    setup_logger()
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
