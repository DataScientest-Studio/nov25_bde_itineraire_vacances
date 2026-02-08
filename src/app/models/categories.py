from pydantic import BaseModel
from typing import List


class MainCategoryResponse(BaseModel):
    main_categories: List[str]

class SubCategoryRequest(BaseModel):
    main_categories: List[str]

class SubCategoryResponse(BaseModel):
    sub_categories: List[str]
