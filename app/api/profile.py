from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.security import current_user, role_names
from app.models.entities import UserAccount
from app.schemas.api import ProfileAuthority, ProfileResponse

router = APIRouter(tags=["profile"])


@router.get("/api/profile", response_model=ProfileResponse)
async def profile(user: Annotated[UserAccount, Depends(current_user)]) -> ProfileResponse:
    return ProfileResponse(
        id=user.id,
        username=user.username,
        displayName=user.display_name,
        roles=[ProfileAuthority(authority=role) for role in sorted(role_names(user))],
    )
