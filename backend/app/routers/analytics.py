"""Analytics endpoints with JWT authentication."""

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.config import settings
from app.schemas.analytics import LoginRequest, LoginResponse
from app.services.analytics_service import get_analytics_data

router = APIRouter(tags=["analytics"])
security = HTTPBearer()

ALGORITHM = "HS256"


def _create_token(username: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expiry_minutes)
    return jwt.encode(
        {"sub": username, "exp": expire},
        settings.jwt_secret,
        algorithm=ALGORITHM,
    )


def _verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    try:
        payload = jwt.decode(credentials.credentials, settings.jwt_secret, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token.")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")


@router.post("/analytics/login")
async def analytics_login(body: LoginRequest):
    if body.username == settings.analytics_username and body.password == settings.analytics_password:
        return LoginResponse(token=_create_token(body.username))
    raise HTTPException(status_code=401, detail="Invalid credentials.")


@router.get("/analytics/data")
async def analytics_data(days: int = 30, _user: str = Depends(_verify_token)):
    return get_analytics_data(days)
