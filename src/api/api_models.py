from typing import List
from pydantic import BaseModel, Field


class CategoriesRequest(BaseModel):
    categories_list: List[str]

class ItineraryRequest(BaseModel):
    sub_categories : List[str]
    longitude : float = Field(..., ge= -180, le= 180)
    latitude : float = Field(..., ge= -90, le= 90)
    radius : int = Field(3000, ge=1000, le= 30000)
    num_days : int = Field(1, ge=1, le= 10)
    mobility_mean : str
    
