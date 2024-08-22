from typing import List
from pydantic import BaseModel


class OpenAITfIdfRequestForm(BaseModel):
    userRequestText: str
