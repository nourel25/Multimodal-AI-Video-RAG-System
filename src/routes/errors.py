import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from models.enums import ResponseSignal
from stores.llm import LLMError

logger = logging.getLogger(__name__)


class AppError(Exception):
    def __init__(self, status_code: int, signal: ResponseSignal, detail: str | None = None):
        super().__init__(detail or signal.value)
        self.status_code = status_code
        self.signal = signal
        self.detail = detail


def _error(status_code: int, signal: ResponseSignal, detail=None) -> JSONResponse:
    return JSONResponse(status_code=status_code,
                        content={"signal": signal.value, "detail": detail})


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError):
        return _error(exc.status_code, exc.signal, exc.detail)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError):
        details = [
            {"loc": list(err.get("loc", [])), "msg": err.get("msg"), "type": err.get("type")}
            for err in exc.errors()
        ]
        return _error(status.HTTP_422_UNPROCESSABLE_CONTENT, ResponseSignal.VALIDATION_ERROR,
                      details)

    @app.exception_handler(LLMError)
    async def llm_error_handler(_: Request, exc: LLMError):
        logger.error("LLM provider error: %s", exc)
        return _error(status.HTTP_502_BAD_GATEWAY, ResponseSignal.PROVIDER_ERROR, str(exc))

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_: Request, exc: Exception):
        logger.exception("Unhandled error", exc_info=exc)
        return _error(status.HTTP_500_INTERNAL_SERVER_ERROR, ResponseSignal.INTERNAL_ERROR)
