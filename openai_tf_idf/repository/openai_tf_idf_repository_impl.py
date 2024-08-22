import os
import httpx
import openai
import faiss
import numpy as np


from dotenv import load_dotenv
from fastapi import HTTPException
from openai_tf_idf.repository.openai_tf_idf_repository import OpenAITfIdfRepository

load_dotenv()

openaiApiKey = os.getenv('OPENAI_API_KEY')
if not openaiApiKey:
    raise ValueError('API Key가 준비되어 있지 않습니다!')

class OpenAITfIdfRepositoryImpl(OpenAITfIdfRepository):

    SIMILARITY_TOP_RANK = 3

    headers = {
        'Authorization': f'Bearer {openaiApiKey}',
        'Content-Type': 'application/json'
    }

    OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"

    async def generateText(self, userSendMessage):
        data = {
            'model': 'gpt-3.5-turbo',
            'messages': [
                {"role": "system", "content": "You are a helpful assitant."},
                {"role": "user", "content": userSendMessage}
            ]
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.OPENAI_CHAT_COMPLETIONS_URL, headers=self.headers, json=data)
                response.raise_for_status()

                return response.json()['choices'][0]['message']['content'].strip()

            except httpx.HTTPStatusError as e:
                print(f"HTTP Error: {str(e)}")
                print(f"Status Code: {e.response.status_code}")
                print(f"Response Text: {e.response.text}")

                raise HTTPException(status_code=e.response.status_code, detail=f"HTTP Error: {e}")

            except (httpx.RequestError, ValueError) as e:
                print(f"Request Error: {e}")
                raise HTTPException(status_code=500, detail=f"Request Error: {e}")

    def openAiBasedEmbedding(self, text):
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )

        print(f"response: {response}")
        return response.data[0].embedding

    def createL2FaissIndex(self, embeddingVectorDimension):
        return faiss.IndexFlatL2(embeddingVectorDimension)

    def similarityAnalysis(self, userRequestPaperTitle, faissIndex):
        embeddingUserRequest = np.array(
            self.openAiBasedEmbedding(userRequestPaperTitle)).astype('float32').reshape(1, -1)
        distanceList, indexList = faissIndex.search(embeddingUserRequest, self.SIMILARITY_TOP_RANK)

        return indexList[0], distanceList[0]