import secrets

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import Depends, Header, Path, Request, status

from core.container import ServiceContainer
from models.db_schemas import User
from models.enums import ResponseSignal
from models.UserModel import UserModel

from .errors import AppError


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


def verify_api_key(x_api_key: str | None = Header(None),
                   container: ServiceContainer = Depends(get_container)) -> None:
    expected = container.settings.API_KEY
    if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
        raise AppError(status.HTTP_401_UNAUTHORIZED, ResponseSignal.UNAUTHORIZED,
                       "Missing or invalid X-API-Key header")


UserIdPath = Path(..., pattern=r"^[A-Za-z0-9_]{3,64}$")


async def get_user_or_404(user_id: str = UserIdPath,
                          container: ServiceContainer = Depends(get_container)) -> User:
    user = await UserModel(container.db_client).get_user(user_id)
    if user is None:
        raise AppError(status.HTTP_404_NOT_FOUND, ResponseSignal.USER_NOT_FOUND,
                       f"User {user_id!r} has no videos yet")
    return user


def parse_object_id(value: str, signal: ResponseSignal) -> ObjectId:
    try:
        return ObjectId(value)
    except (InvalidId, TypeError) as exc:
        raise AppError(status.HTTP_404_NOT_FOUND, signal, f"Unknown id {value!r}") from exc
