from fastapi import APIRouter, HTTPException

from .. import share
from ..schemas import HostnameIn, ShareStatusOut

router = APIRouter(prefix="/api/share", tags=["share"])


@router.get("/status", response_model=ShareStatusOut)
async def get_status():
    return ShareStatusOut(**share.get_share_status())


@router.post("/upnp/enable", response_model=ShareStatusOut)
async def enable_upnp():
    share.enable_upnp()
    return ShareStatusOut(**share.get_share_status())


@router.post("/upnp/disable", response_model=ShareStatusOut)
async def disable_upnp():
    share.disable_upnp()
    return ShareStatusOut(**share.get_share_status())


@router.post("/hostname", response_model=ShareStatusOut)
async def set_hostname(body: HostnameIn):
    try:
        share.set_hostname(body.hostname)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ShareStatusOut(**share.get_share_status())
