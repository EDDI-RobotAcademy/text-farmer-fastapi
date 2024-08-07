from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from tf_idf_bow.service.tf_idf_bow_service_impl import TfIdfBowServiceImpl

tfIdfBowRouter = APIRouter()

async def injectTfIdfBowService() -> TfIdfBowServiceImpl:
    return TfIdfBowServiceImpl()

@tfIdfBowRouter.post("/find-similar-answer")
async def findSimilarAnswer(tfIdfBowService: TfIdfBowServiceImpl =
                            Depends(injectTfIdfBowService)):

    print(f"controller -> findSimilarAnswer(): tfIdfBowRequestForm")