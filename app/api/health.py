import logging
from typing import Callable, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.config import get_settings
from app.db.database import engine

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


def _run_check(name: str, check: Callable[[], Dict[str, str]]) -> Dict[str, str]:
    try:
        return {"status": "ok", **check()}
    except Exception as error:
        logger.warning("Health check failed for %s: %s", name, error)
        return {"status": "unavailable"}


@router.get("/health")
def health(request: Request) -> JSONResponse:
    state = request.app.state
    settings = get_settings()

    def check_database() -> Dict[str, str]:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {}

    def check_qdrant() -> Dict[str, str]:
        state.vector_store.ensure_collection()
        return {}

    def check_redis() -> Dict[str, str]:
        state.memory_service.ping()
        return {}

    def check_ollama() -> Dict[str, str]:
        models = state.llm_service.list_models()
        wanted = settings.llm_model
        if not any(model == wanted or model.split(":")[0] == wanted for model in models):
            raise RuntimeError(f"model {wanted} has not been pulled")
        return {"model": wanted}

    def check_embedding_model() -> Dict[str, str]:
        return {"model": state.embedding_service.model_name}

    checks = {
        "database": _run_check("database", check_database),
        "qdrant": _run_check("qdrant", check_qdrant),
        "redis": _run_check("redis", check_redis),
        "ollama": _run_check("ollama", check_ollama),
        "embedding_model": _run_check("embedding_model", check_embedding_model),
    }
    healthy = all(result["status"] == "ok" for result in checks.values())

    return JSONResponse(
        status_code=200 if healthy else 503,
        content={"status": "ok" if healthy else "degraded", "checks": checks},
    )
