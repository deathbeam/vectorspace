import chromadb
import uvicorn
from chromadb import Collection, GetResult
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor
from chromadb.utils import embedding_functions
from fastapi import FastAPI
from pydantic import BaseModel
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler


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
    existing: GetResult = collection.get(include=["metadatas"])

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
        file_contents = open(file).read()
    except:
        return

    print(file)
    collection.add(
        documents=[file_contents],
        metadatas=[{"mtime": os.path.getmtime(file)}],
        ids=[file],
    )


class Watch(BaseModel):
    dir: str


class Query(Watch):
    text: str
    max: int


class FileChangeHandler(PatternMatchingEventHandler):
    def __init__(self, collection: Collection):
        super().__init__(ignore_patterns=["*/.git/*"], ignore_directories=True)
        self.collection = collection

    def on_modified(self, event):
        if event.is_directory:
            return

        read_file(self.collection, event.src_path)


app = FastAPI(title="vectorspace")
db = chromadb.PersistentClient(path=os.getenv("HOME") + "/.cache/chromadb")
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="multi-qa-MiniLM-L6-cos-v1"
)
observer = Observer()
observer.start()
executor = ThreadPoolExecutor(50)
watches = {}


def col(dir):
    return db.get_or_create_collection(
        name=dir.replace("/", ""),
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function,  # type: ignore
    )


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
    return watch


@app.post("/count")
def count(watch: Watch):
    return col(watch.dir).count()


@app.post("/query")
def query(query: Query):
    return col(query.dir).query(
        query_texts=[query.text],
        n_results=query.max,
    )

def main():
    uvicorn.run(app)

if __name__ == "__main__":
    main()
