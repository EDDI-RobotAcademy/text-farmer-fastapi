import os
import sys

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from openai_tf_idf.service.openai_tf_idf_service_impl import OpenAITfIdfServiceImpl
from user_defined_queue.repository.user_defined_queue_repository_impl import UserDefinedQueueRepositoryImpl

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'template'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'template', 'include', 'socket_server'))

from template.include.socket_server.utility.color_print import ColorPrinter

openAITfIdfRouter = APIRouter()

async def injectOpenAITfIdfBowService() -> OpenAITfIdfServiceImpl:
    return OpenAITfIdfServiceImpl(UserDefinedQueueRepositoryImpl.getInstance())


@openAITfIdfRouter.post("/find_similar/")
async def find_similar(openAITfIdfService: OpenAITfIdfServiceImpl =
                    Depends(injectOpenAITfIdfBowService)):

    ColorPrinter.print_important_message("OpenAI-find_similar()")

    generatedAnswers = openAITfIdfService.textSimilarityAnalysis()

    return JSONResponse(content=generatedAnswers, status_code=status.HTTP_200_OK)