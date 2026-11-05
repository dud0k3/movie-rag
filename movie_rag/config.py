from dataclasses import dataclass
from pathlib import Path
from importlib.util import find_spec
import os

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    db_path: Path = Path(os.getenv("MOVIE_RAG_DB", str(ROOT / "data" / "movies.sqlite3")))
    vector_path: Path = Path(os.getenv("MOVIE_RAG_VECTORS", str(ROOT / "data" / "vectors.npz")))
    tmdb_api_key: str = os.getenv("TMDB_API_KEY", "").strip()
    tmdb_token: str = os.getenv("TMDB_READ_ACCESS_TOKEN", "").strip()
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    offline: bool = os.getenv("MOVIE_RAG_OFFLINE", "").lower() in {"1", "true", "yes"}
    wikimedia_contact_url: str = os.getenv("WIKIMEDIA_CONTACT_URL", "").strip()

    @property
    def has_tmdb_auth(self) -> bool:
        return bool(self.tmdb_api_key or self.tmdb_token)

    @property
    def demo_mode(self) -> bool:
        return not self.has_tmdb_auth

    @property
    def semantic_enabled(self) -> bool:
        return not self.offline and find_spec("sentence_transformers") is not None


settings = Settings()
