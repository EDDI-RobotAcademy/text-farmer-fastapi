import os
import faiss
import numpy as np
import openai

from dotenv import load_dotenv

from openai_tf_idf.repository.openai_tf_idf_repository import OpenAITfIdfRepository


load_dotenv()

openaiApiKey = os.getenv('OPENAI_API_KEY')
if not openaiApiKey:
    raise ValueError('API Key가 준비되어 있지 않습니다!')

openai.api_key = openaiApiKey

class OpenAITfIdfRepositoryImpl(OpenAITfIdfRepository):
    SIMILARITY_TOP_RANK = 3

    headers = {
        'Authorization': f'Bearer {openaiApiKey}',
        'Content-Type': 'application/json'
    }

    OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"


    def openAiBasedEmbedding(self, text):
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )

        print(f"response: {response}")
        return response['data'][0]['embedding']

    def createL2FaissIndex(self, embeddingVectorDimension):
        return faiss.IndexFlatL2(embeddingVectorDimension)

    def similarityAnalysis(self, userQuestion, faissIndex):
        embeddingUserQuestion = np.array(
            self.openAiBasedEmbedding(userQuestion)).astype('float32').reshape(1, -1)
        distanceList, indexList = faissIndex.search(embeddingUserQuestion, self.SIMILARITY_TOP_RANK)

        return indexList[0], distanceList[0]



