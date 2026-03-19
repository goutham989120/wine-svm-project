from pydantic import BaseModel
from typing import List

class WineInput(BaseModel):
    features: List[float]