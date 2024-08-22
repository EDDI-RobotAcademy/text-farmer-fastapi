from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from openai_tf_idf.controller.request_form.openai_paper_similarity_analysis_request_form import \
    OpenAIPaperSimilarityAnalysisRequestForm
from openai_tf_idf.service.openai_tf_idf_service_impl import OpenAITfIdfServiceImpl

openAITfIdfRouter = APIRouter()

async def injectOpenAITfIdfService() -> OpenAITfIdfServiceImpl:
    return OpenAITfIdfServiceImpl()

@openAITfIdfRouter.post("/openai-similarity-analysis")
async def textSimilarityAnalysisWithOpenAI(
        openAIPaperSimilarityAnalysisRequestForm: OpenAIPaperSimilarityAnalysisRequestForm,
        openAITfIdfService: OpenAITfIdfServiceImpl =
        Depends(injectOpenAITfIdfService)):

    print(f"controller -> textSimilarityAnalysisWithOpenAI(): "
          f"openAIPaperSimilarityAnalysisRequestForm: {openAIPaperSimilarityAnalysisRequestForm}")

    analyzedSimilarityText = await openAITfIdfService.textSimilarityAnalysis(
        openAIPaperSimilarityAnalysisRequestForm.paperTitleList,
        openAIPaperSimilarityAnalysisRequestForm.userRequestPaperTitle)

    return JSONResponse(
        content={"result": jsonable_encoder(analyzedSimilarityText)},
        status_code=status.HTTP_200_OK)