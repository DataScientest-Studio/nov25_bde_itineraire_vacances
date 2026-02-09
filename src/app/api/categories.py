from fastapi import APIRouter, Depends
from app.dependencies import get_db
from app.services.category_service import category_service
from app.models.categories import MainCategoryResponse, SubCategoryRequest, SubCategoryResponse


router = APIRouter(tags=["categories"])

@router.get("/main_categories", response_model=MainCategoryResponse)
def get_main_categories(db=Depends(get_db)):
    cats = category_service.get_main_categories(db)
    return MainCategoryResponse(main_categories=cats)

@router.post("/sub_categories", response_model=SubCategoryResponse)
def get_sub_categories(req: SubCategoryRequest, db=Depends(get_db)):
    subs = category_service.get_sub_categories(db, req.main_categories)
    return SubCategoryResponse(sub_categories=subs)
