from fastapi import APIRouter

from .. import share
from ..schemas import ShareStatusOut

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
