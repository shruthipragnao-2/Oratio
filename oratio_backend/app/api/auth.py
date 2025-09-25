from dataclasses import dataclass
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field


router = APIRouter(prefix="/auth", tags=["auth"])

# Minimal in-memory store for demo purposes
_users: Dict[str, str] = {}


class SignupPayload(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)


class LoginPayload(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


security = HTTPBearer(auto_error=False)


def get_current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> str:
    if creds is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = creds.credentials
    # For demo: token is the email itself
    if token not in _users:
        raise HTTPException(status_code=401, detail="Invalid token")
    return token


@router.post("/signup", response_model=AuthResponse)
def signup(payload: SignupPayload) -> AuthResponse:
    if payload.email in _users:
        raise HTTPException(status_code=400, detail="User already exists")
    # WARNING: For demo only. Do not store plain-text passwords.
    _users[payload.email] = payload.password
    return AuthResponse(access_token=payload.email)


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginPayload) -> AuthResponse:
    if _users.get(payload.email) != payload.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return AuthResponse(access_token=payload.email)


class MeResponse(BaseModel):
    email: EmailStr


@router.get("/me", response_model=MeResponse)
def me(user_email: str = Depends(get_current_user)) -> MeResponse:
    return MeResponse(email=user_email)


