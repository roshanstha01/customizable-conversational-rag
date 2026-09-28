import logging
from contextlib import contextmanager
from typing import Iterator, Optional, Tuple, Type

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ServiceUnavailableError(Exception):
    """Raised when a backing service (Qdrant, Redis, Ollama) cannot be reached."""

    def __init__(self, service: str, message: Optional[str] = None) -> None:
        self.service = service
        self.message = message or f"{service} is unavailable. Please try again later."
        super().__init__(self.message)


@contextmanager
def unavailable_on(errors: Tuple[Type[BaseException], ...], service: str) -> Iterator[None]:
    """Translate low-level connection errors into ServiceUnavailableError."""
    try:
        yield
    except errors as error:
        logger.warning("%s unavailable: %s", service, error)
        raise ServiceUnavailableError(service) from error


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceUnavailableError)
    async def service_unavailable_handler(request: Request, error: ServiceUnavailableError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": error.message},
        )

    @app.middleware("http")
    async def catch_unhandled_errors(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception:
            # Log the full traceback server-side; never leak exception text to clients.
            logger.exception("Unhandled error on %s %s", request.method, request.url.path)
            return JSONResponse(status_code=500, content={"detail": "Internal server error"})
