from fastapi import APIRouter
router = APIRouter()

@router.get("/performance")
def get_model_performance():
    return {"message": "Model not trained yet."}

@router.get("/importance")
def get_feature_importance():
    return {"features": []}
