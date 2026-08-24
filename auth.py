from fastapi import Depends, Header, HTTPException

from config import Settings, get_settings


async def require_api_key(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="API key missing.")

    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key.")