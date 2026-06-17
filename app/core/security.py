import secrets
from typing import Annotated

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session
from app.models.entities import UserAccount

security = HTTPBasic()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


async def current_user(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> UserAccount:
    result = await session.scalars(
        select(UserAccount).where(UserAccount.username == credentials.username, UserAccount.enabled.is_(True))
    )
    user = result.first()
    if not user or not verify_password(credentials.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    if not secrets.compare_digest(user.username, credentials.username):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return user


def role_names(user: UserAccount) -> set[str]:
    return {role.role for role in user.roles}


async def require_admin(user: Annotated[UserAccount, Depends(current_user)]) -> UserAccount:
    if "ROLE_ADMIN" not in role_names(user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")
    return user


async def require_student(user: Annotated[UserAccount, Depends(current_user)]) -> UserAccount:
    if "ROLE_ADMIN" in role_names(user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="管理员账号只能查看后台记录，不能发起学生对话。")
    return user
