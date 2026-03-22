from fastapi import APIRouter
router = APIRouter()

@router.get("/")
def list_fighters():
    return {"fighters": [], "total": 0}

@router.get("/{fighter_id}")
def get_fighter(fighter_id: str):
    return {"fighter_id": fighter_id, "message": "Fighter data not loaded yet."}
