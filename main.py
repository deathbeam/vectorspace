import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

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

        file_contents = open(event.src_path).read()
        self.collection.add(
            documents = [ file_contents ],
            ids = [ event.src_path ],
        )

app = FastAPI()
db = chromadb.PersistentClient(path="~/.cache/chromadb")
observer = Observer()
observer.start()
watches = {}

def col(dir):
    return db.get_or_create_collection(dir.replace("/", ""))

@app.post("/start")
def start(watch: Watch):
    watches[watch.dir] = observer.schedule(FileChangeHandler(col(watch.dir)), watch.dir, recursive=True)
    return watch

@app.post("/stop")
def stop(watch: Watch):
    observer.unschedule(watches[watch.dir])
    return watch

@app.post("/query")
def query(query: Query):
    return col(query.dir).query(
        query_texts = [ query.text ],
        n_results = query.max,
    )
