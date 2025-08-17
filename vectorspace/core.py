import threading
import chromadb
from chromadb import Collection, GetResult, QueryResult
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor
from chromadb.utils import embedding_functions
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import time
from typing import List, Dict, Any


class VectorspaceCore:
    def __init__(self, cache_path: str|None = None):
        if cache_path is None:
            cache_path = os.getenv("HOME") + "/.cache/chromadb"

        os.environ["ANONYMIZED_TELEMETRY"] = "False"

        self.db = chromadb.PersistentClient(path=cache_path)
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        self.observer = Observer()
        self.observer.start()
        self.executor = ThreadPoolExecutor(50)
        self.watches = {}
        self.watch_last_used = {}
        self.watch_cleanup_interval = 300  # 5 minutes

        # Start cleanup thread
        threading.Thread(target=self._cleanup_inactive_watches, daemon=True).start()

    def col(self, dir_path: str) -> Collection:
        self.watch_last_used[dir_path] = time.time()
        return self.db.get_or_create_collection(
            name=dir_path.replace("/", "").replace("\\", ""),
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_function, # type: ignore
        )

    def all_files(self, collection: Collection, dir_path: str) -> List[str]:
        files = (
            subprocess.run(
                ["git", "-C", dir_path, "ls-files", "--full-name"], capture_output=True
            )
            .stdout.decode()
            .split("\n")
        )

        files = [os.path.join(dir_path, file) for file in files if "/.git" not in file]
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

            existing_mtime = float(existing_metadata.get("mtime", 0)) # type: ignore
            mtime = os.path.getmtime(file)
            if mtime > existing_mtime:
                out.append(file)

        return out

    def read_file(self, collection: Collection, file_path: str):
        try:
            if os.path.getsize(file_path) > 1024 * 1024:
                return

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                file_contents = f.read(1024)
                if "\0" in file_contents:
                    return
                file_contents += f.read()

            print(f"Indexing: {file_path}")
            collection.upsert(
                documents=[file_contents],
                metadatas=[{"mtime": os.path.getmtime(file_path)}],
                ids=[file_path],
            )
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            return

    def start_watching(self, dir_path: str) -> Dict[str, Any]:
        collection = self.col(dir_path)

        if dir_path in self.watches:
            return {
                "dir": dir_path,
                "files": collection.count(),
            }

        files = self.all_files(collection, dir_path)
        for file in files:
            self.executor.submit(self.read_file, collection, file)

        self.watches[dir_path] = self.observer.schedule(
            FileChangeHandler(collection), dir_path, recursive=True
        )

        return {
            "dir": dir_path,
            "files": len(files),
        }

    def stop_watching(self, dir_path: str):
        if dir_path not in self.watches:
            return

        self.observer.unschedule(self.watches[dir_path])
        del self.watches[dir_path]
        if dir_path in self.watch_last_used:
            del self.watch_last_used[dir_path]

    def count_files(self, dir_path: str) -> int:
        return self.col(dir_path).count()

    def query(self, dir_path: str, text: str, max_results: int = 10) -> List[Dict[str, Any]]:
        self.start_watching(dir_path)

        response: QueryResult = self.col(dir_path).query(
            query_texts=[text],
            n_results=max_results,
        )

        out = []
        for i in range(len(response["ids"][0])):
            id = response["ids"][0][i]
            document = response["documents"][0][i]
            distance = response["distances"][0][i]

            out.append({
                "filename": id,
                "content": document,
                "score": max(0.0, min(1.0, 1.0 - (distance / 2.0))),
            })

        return out

    def _cleanup_inactive_watches(self):
        while True:
            time.sleep(60)
            current_time = time.time()
            inactive_dirs = []
            for dir_path, last_used in list(self.watch_last_used.items()):
                if current_time - last_used > self.watch_cleanup_interval:
                    inactive_dirs.append(dir_path)

            for dir_path in inactive_dirs:
                print(f"Auto-stopping inactive watch for: {dir_path}")
                self.stop_watching(dir_path)


class FileChangeHandler(PatternMatchingEventHandler):
    def __init__(self, collection: Collection):
        super().__init__(ignore_patterns=["*/.git/*"], ignore_directories=True)
        self.collection = collection

    def on_modified(self, event):
        if event.is_directory:
            return
        self._read_file(str(event.src_path))

    def on_created(self, event):
        if event.is_directory:
            return
        self._read_file(str(event.src_path))

    def on_deleted(self, event):
        if event.is_directory:
            return
        print(f"File deleted: {event.src_path}")
        self.collection.delete(ids=[str(event.src_path)])

    def _read_file(self, file_path: str):
        try:
            if os.path.getsize(file_path) > 1024 * 1024:
                return

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                file_contents = f.read(1024)
                if "\0" in file_contents:
                    return
                file_contents += f.read()

            print(f"Indexing: {file_path}")
            self.collection.upsert(
                documents=[file_contents],
                metadatas=[{"mtime": os.path.getmtime(file_path)}],
                ids=[file_path],
            )
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
