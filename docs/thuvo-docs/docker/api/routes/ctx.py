from fastapi import APIRouter
from data.loaders import load_datatourisme_df, load_airbnb_df, load_tripadvisor_df
from schemas.context import ContextResponse

router = APIRouter(tags=["Context"])

@router.get("/ctx/dt", response_model=ContextResponse)
def ctx_dt(limit: int = 5000):
    df = load_datatourisme_df(limit=limit)
    records = df.to_dict(orient="records")
    return {"items": records, "count": len(records)}


@router.get("/ctx/ab", response_model=ContextResponse)
def ctx_ab(limit: int = 5000):
    df = load_airbnb_df(limit=limit)
    records = df.to_dict(orient="records")
    return {"items": records, "count": len(records)}


@router.get("/ctx/ta", response_model=ContextResponse)
def ctx_ta(limit: int = 5000):
    df = load_tripadvisor_df(limit=limit)
    records = df.to_dict(orient="records")
    return {"items": records, "count": len(records)}