import threading
from typing import List
import chromadb
from chromadb.api.types import IncludeEnum
import uvicorn
from chromadb import Collection, GetResult, QueryResult
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor
from chromadb.utils import embedding_functions
from fastapi import FastAPI
from pydantic import BaseModel
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import time


def all_files(collection: Collection, dir):
    files = (
        subprocess.run(
            ["git", "-C", dir, "ls-files", "--full-name"], capture_output=True
        )
        .stdout.decode()
        .split("\n")
    )

    files = [os.path.join(dir, file) for file in files if "/.git" not in file]
    files = [file for file in files if os.path.isfile(file)]
    existing: GetResult = collection.get(include=[IncludeEnum.metadatas])

    out = []
    for file in files:
        if file not in existing["ids"]:
            out.append(file)
            continue

        existing_file_index = existing["ids"].index(file)
        existing_metadata = existing["metadatas"][existing_file_index]
        if not existing_metadata:
            out.append(file)
            continue

        existing_mtime = float(existing_metadata.get("mtime", 0))
        mtime = os.path.getmtime(file)
        if mtime > existing_mtime:
            out.append(file)

    return out


def read_file(collection: Collection, file: str):
    try:
        # Skip files larger than 1MB to avoid performance issues
        if os.path.getsize(file) > 1024 * 1024:
            print(f"Skipping large file: {file}")
            return

        # Skip binary files by checking for null bytes
        with open(file, "rb") as f:
            content = f.read(1024)  # Read first 1KB to check
            if b"\x00" in content:
                print(f"Skipping binary file: {file}")
                return

        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            file_contents = f.read()

        print(f"Indexing: {file}")
        collection.add(
            documents=[file_contents],
            metadatas=[{"mtime": os.path.getmtime(file)}],
            ids=[file],
        )
    except Exception as e:
        print(f"Error processing file {file}: {str(e)}")
        return


class Watch(BaseModel):
    dir: str


class Query(Watch):
    text: str
    max: int


class QueryData(BaseModel):
    filename: str
    content: str
    score: float


class FileChangeHandler(PatternMatchingEventHandler):
    def __init__(self, collection: Collection):
        super().__init__(ignore_patterns=["*/.git/*"], ignore_directories=True)
        self.collection = collection

    def on_modified(self, event):
        if event.is_directory:
            return

        read_file(self.collection, str(event.src_path))


app = FastAPI(title="vectorspace")
db = chromadb.PersistentClient(path=os.getenv("HOME") + "/.cache/chromadb")
embedding_function = embedding_functions.DefaultEmbeddingFunction()
observer = Observer()
observer.start()
executor = ThreadPoolExecutor(50)
watches = {}
watch_last_used = {}  # Track when each watch was last used
watch_cleanup_interval = 300  # 5 minutes in seconds


def col(dir):
    watch_last_used[dir] = time.time()
    return db.get_or_create_collection(
        name=dir.replace("/", ""),
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,  # type: ignore
    )


def cleanup_inactive_watches():
    while True:
        time.sleep(60)  # Check every minute
        current_time = time.time()
        inactive_dirs = []
        for dir_path, last_used in list(watch_last_used.items()):
            if current_time - last_used > watch_cleanup_interval:
                inactive_dirs.append(dir_path)

        for dir_path in inactive_dirs:
            print(f"Auto-stopping inactive watch for: {dir_path}")
            stop(Watch(dir=dir_path))


threading.Thread(target=cleanup_inactive_watches, daemon=True).start()


@app.post("/start")
def start(watch: Watch):
    collection = col(watch.dir)

    if watch.dir in watches:
        return {
            "dir": watch.dir,
            "files": collection.count(),
        }

    files = all_files(collection, watch.dir)
    for file in files:
        executor.submit(read_file, collection, file)

    watches[watch.dir] = observer.schedule(
        FileChangeHandler(collection), watch.dir, recursive=True
    )

    return {
        "dir": watch.dir,
        "files": len(files),
    }


@app.post("/stop")
def stop(watch: Watch):
    if watch.dir not in watches:
        return watch

    observer.unschedule(watches[watch.dir])
    del watches[watch.dir]
    if watch.dir in watch_last_used:
        del watch_last_used[watch.dir]
    return watch


@app.post("/count")
def count(watch: Watch) -> int:
    return col(watch.dir).count()


@app.post("/query")
def query(query: Query) -> List[QueryData]:
    start(query)

    response: QueryResult = col(query.dir).query(
        query_texts=[query.text],
        n_results=query.max,
    )

    out = []

    for i in range(len(response["ids"][0])):
        id = response["ids"][0][i]
        document = response["documents"][0][i]
        distance = response["distances"][0][i]

        out.append(
            QueryData(
                filename=id,
                content=document,
                score=max(0.0, min(1.0, 1.0 - (distance / 2.0))),
            )
        )

    return out


def main():
    uvicorn.run(app)


if __name__ == "__main__":
    main()
