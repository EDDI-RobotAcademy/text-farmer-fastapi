from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from openai_tf_idf.controller.request_form.openai_tf_idf_request_form import \
    OpenAITfIdfRequestForm
from openai_tf_idf.service.openai_tf_idf_service_impl import OpenAITfIdfServiceImpl

openAITfIdfRouter = APIRouter()

async def injectOpenAITfIdfService() -> OpenAITfIdfServiceImpl:
    return OpenAITfIdfServiceImpl()

@openAITfIdfRouter.post("/openai-similarity-analysis")
async def textSimilarityAnalysisWithOpenAI(
        openAITfIdfRequestForm: OpenAITfIdfRequestForm,
        openAITfIdfService: OpenAITfIdfServiceImpl =
        Depends(injectOpenAITfIdfService)):

    print(f"controller -> textSimilarityAnalysisWithOpenAI(): "
          f"openAITfIdfRequestForm: {openAITfIdfRequestForm}")

    try:
        analyzedSimilarityText = await openAITfIdfService.findBestResponse(
            openAITfIdfRequestForm.userRequestText
        )

        return JSONResponse(
            content={"result": jsonable_encoder(analyzedSimilarityText)},
            status_code=status.HTTP_200_OK
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
