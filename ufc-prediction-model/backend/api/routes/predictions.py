from fastapi import APIRouter
router = APIRouter()

@router.get("/upcoming")
def get_upcoming_predictions():
    return {"message": "No predictions available yet. Train the model first."}
