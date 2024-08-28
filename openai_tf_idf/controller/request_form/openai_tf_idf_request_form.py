from pydantic import BaseModel
from typing import List

class AddTextsRequest(BaseModel):
    texts: List[str]

class FindSimilarRequest(BaseModel):
    userQuestion: str
    top_k: int = 3
