import chromadb
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor
from chromadb.utils import embedding_functions
from fastapi import FastAPI
from pydantic import BaseModel
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


def git_files(dir):
    files = (
        subprocess.run(
            ["git", "-C", dir, "ls-files", "--full-name"], capture_output=True
        )
        .stdout.decode()
        .split("\n")
    )
    return [os.path.join(dir, file) for file in files]


def read_file(collection, file):
    if "/.git" in file:
        return

    if not os.path.isfile(file):
        return
    try:
        file_contents = open(file).read()
    except:
        return

    print(file)
    collection.add(
        documents=[file_contents],
        ids=[file],
    )


class Watch(BaseModel):
    dir: str


class Query(Watch):
    text: str
    max: int


class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, collection):
        self.collection = collection

    def on_modified(self, event):
        if event.is_directory:
            return

        read_file(self.collection, event.src_path)


app = FastAPI()
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

    files = git_files(watch.dir)
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
