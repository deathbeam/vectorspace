import logging
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
from .chunking import chunk_file

logger = logging.getLogger(__name__)


def remove_file(collection: Collection, file_path: str):
    try:
        logger.info(f"Removing file from index: {file_path}")
        collection.delete(where={"filename": file_path})
    except Exception as e:
        logger.error(f"Error removing file {file_path}: {str(e)}")
        return


def read_file(collection: Collection, file_path: str):
    try:
        if os.path.getsize(file_path) > 1024 * 1024:
            return

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            file_contents = f.read(1024)
            if "\0" in file_contents:
                return
            file_contents += f.read()

        remove_file(collection, file_path)

        logger.info(f"Indexing: {file_path}")
        chunks = chunk_file(file_path, file_contents)
        mtime = os.path.getmtime(file_path)
        documents = [chunk["body"] for chunk in chunks]
        metadatas = [{**chunk["metadata"], "mtime": mtime, "filename": file_path} for chunk in chunks]
        ids = [
            f"{file_path}:{chunk['metadata'].get('start_row', 0)}:{chunk['metadata'].get('start_col', 0)}:{chunk['metadata'].get('end_row', 0)}:{chunk['metadata'].get('end_col', 0)}"
            for chunk in chunks
        ]
        collection.upsert(
            documents=documents,
            metadatas=metadatas,  # type: ignore
            ids=ids,
        )
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return


def sync_files(collection: Collection, dir_path: str) -> List[str]:
    files = (
        subprocess.run(["git", "-C", dir_path, "ls-files", "--full-name"], capture_output=True)
        .stdout.decode()
        .split("\n")
    )

    files = [os.path.join(dir_path, file) for file in files if "/.git" not in file]
    files = [file for file in files if os.path.isfile(file)]
    existing: GetResult = collection.get(include=["metadatas"])

    out = []
    for file in files:
        # Find all chunk IDs for this file
        file_chunk_indices = [i for i, id in enumerate(existing["ids"]) if id.startswith(file)]
        if not file_chunk_indices:
            out.append(file)
            continue

        # Check mtimes for all chunks
        file_mtime = os.path.getmtime(file)
        for index in file_chunk_indices:
            existing_metadata = existing["metadatas"][index]
            if not existing_metadata:
                out.append(file)
                break

            existing_mtime = float(existing_metadata.get("mtime", 0))  # type: ignore
            if file_mtime > existing_mtime:
                out.append(file)
                break

    return out


class VectorspaceCore:
    def __init__(self, cache_path: str | None = None):
        if cache_path is None:
            cache_path = os.getenv("HOME") + "/.cache/vectorspace"

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
            embedding_function=self.embedding_function,  # type: ignore
        )

    def start_watching(self, dir_path: str) -> Dict[str, Any]:
        collection = self.col(dir_path)

        if dir_path in self.watches:
            return {
                "dir": dir_path,
                "files": collection.count(),
            }

        files = sync_files(collection, dir_path)
        for file in files:
            self.executor.submit(read_file, collection, file)

        self.watches[dir_path] = self.observer.schedule(FileChangeHandler(collection), dir_path, recursive=True)

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

    def clear(self, dir_path: str):
        collection = self.col(dir_path)
        self.db.delete_collection(name=collection.name)

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
            document = response["documents"][0][i]
            distance = response["distances"][0][i]
            metadata = response["metadatas"][0][i]

            out.append(
                {
                    "filename": metadata.get("filename"),
                    "content": document,
                    "score": max(0.0, min(1.0, 1.0 - (distance / 2.0))),
                    "start_row": metadata.get("start_row", 0),
                    "end_row": metadata.get("end_row", 0),
                    "start_col": metadata.get("start_col", 0),
                    "end_col": metadata.get("end_col", 0),
                }
            )

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
                logger.info(f"Auto-stopping inactive watch for: {dir_path}")
                self.stop_watching(dir_path)


class FileChangeHandler(PatternMatchingEventHandler):
    def __init__(self, collection: Collection):
        super().__init__(ignore_patterns=["*/.git/*"], ignore_directories=True)
        self.collection = collection

    def on_modified(self, event):
        if event.is_directory:
            return
        read_file(self.collection, str(event.src_path))

    def on_created(self, event):
        if event.is_directory:
            return
        read_file(self.collection, str(event.src_path))

    def on_deleted(self, event):
        if event.is_directory:
            return
        remove_file(self.collection, str(event.src_path))
