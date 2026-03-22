from fastapi import APIRouter
router = APIRouter()

@router.get("/history")
def get_event_history():
    return {"events": []}
