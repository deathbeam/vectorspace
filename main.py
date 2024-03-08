import chromadb
import subprocess
import os
import time
from chromadb.utils import embedding_functions
from fastapi import FastAPI
from pydantic import BaseModel
from watchdog.observers import Observer
from watchdog.events import FileSystemEvent, FileSystemEventHandler

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

        if "/.git" in event.src_path:
            return

        try:
            file_contents = open(event.src_path).read()
        except:
            return

        print(event.src_path)
        self.collection.add(
            documents = [ file_contents ],
            ids = [ event.src_path ],
        )

app = FastAPI()
db = chromadb.PersistentClient(path=os.getenv("HOME") + "/.cache/chromadb")
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="multi-qa-MiniLM-L6-cos-v1")
observer = Observer()
observer.start()
watches = {}

def git_files(dir):
    files = subprocess.run(
        ["git", "-C", dir, "ls-files", "--full-name"],
        capture_output=True
    ).stdout.decode().split("\n")
    files = [os.path.join(dir, file) for file in files]
    for file in files:
        print(file)
    return [file for file in files if os.path.isfile(file)]

def col(dir):
    return db.get_or_create_collection(
        name=dir.replace("/", ""),
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function
    )

@app.post("/start")
def start(watch: Watch):
    start_time = time.time()
    handler = FileChangeHandler(col(watch.dir))
    files = git_files(watch.dir)
    for file in files:
        handler.on_modified(FileSystemEvent(src_path=file))
    end = time.time()
    duration = end - start_time
    watches[watch.dir] = observer.schedule(handler, watch.dir, recursive=True)
    return {
        'dir': watch.dir,
        'files': len(files),
        'duration': duration,
    }

@app.post("/stop")
def stop(watch: Watch):
    observer.unschedule(watches[watch.dir])
    return watch

@app.post("/count")
def count(watch: Watch):
    return col(watch.dir).count()

@app.post("/query")
def query(query: Query):
    return col(query.dir).query(
        query_texts = [ query.text ],
        n_results = query.max,
    )
