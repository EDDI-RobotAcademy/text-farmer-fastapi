from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv
import os

from openai_tf_idf.controller.request_form.openai_tf_idf_request_form import FindSimilarRequest
from openai_tf_idf.service.openai_tf_idf_service_impl import OpenAITfIdfServiceImpl

openAITfIdfRouter = APIRouter()

load_dotenv()

# Service를 초기화할 때 필요한 파일 경로를 환경 변수에서 가져옵니다.
embedding_pickle_path = r"C:\TeamProject\SK-Networks-AI-1\TF\text-farmer-fastapi\app\assets\예방_Embedded_answers.pickle"
original_data_path = r"C:\TeamProject\SK-Networks-AI-1\TF\text-farmer-fastapi\app\assets\예방_original_answers.csv"

# 환경 변수 확인
if not embedding_pickle_path or not original_data_path:
    raise ValueError('EMBEDDING_PICKLE_PATH 및 ORIGINAL_DATA_PATH가 설정되지 않았습니다.')

service = OpenAITfIdfServiceImpl()

@openAITfIdfRouter.post("/add_texts/")
async def add_texts(texts: list[str]):
    try:
        service.add_texts_to_index(texts)
        return {"message": "Texts successfully added to index"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@openAITfIdfRouter.post("/find_similar/")
async def find_similar(request: FindSimilarRequest):
    try:
        response = await service.textSimilarityAnalysis(request.userQuestion, request.top_k)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
