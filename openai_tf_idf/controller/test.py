import os
import sys

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from tf_idf_bow.service.tf_idf_bow_service_impl import TfIdfBowServiceImpl
from user_defined_queue.repository.user_defined_queue_repository_impl import UserDefinedQueueRepositoryImpl

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'template'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'template', 'include', 'socket_server'))

from template.include.socket_server.utility.color_print import ColorPrinter

tfIdfBowRouter = APIRouter()

async def injectTfIdfBowService() -> TfIdfBowServiceImpl:
    return TfIdfBowServiceImpl(UserDefinedQueueRepositoryImpl.getInstance())

@tfIdfBowRouter.get("/find-similar-answer")
async def findSimilarAnswer(tfIdfBowService: TfIdfBowServiceImpl =
                            Depends(injectTfIdfBowService)):

    ColorPrinter.print_important_message("findSimilarAnswer()")

    generatedSimilarAnswer = tfIdfBowService.findSimilarAnswerInfo()

    return JSONResponse(content=generatedSimilarAnswer, status_code=status.HTTP_200_OK)