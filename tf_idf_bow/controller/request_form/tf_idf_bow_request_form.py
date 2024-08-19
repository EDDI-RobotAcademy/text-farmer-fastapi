from pydantic import BaseModel

class TfIdfBowRequestForm(BaseModel):
    userSendQuestion: str