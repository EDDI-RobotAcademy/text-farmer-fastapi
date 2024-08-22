from typing import List
import numpy as np

from openai_tf_idf.controller.response_form.openai_tf_idf_response_form import OpenAITfIdfResponseForm
from openai_tf_idf.repository.openai_tf_idf_repository_impl import OpenAITfIdfRepositoryImpl
from openai_tf_idf.service.openai_tf_idf_service import OpenAITfIdfService


class OpenAITfIdfServiceImpl(OpenAITfIdfService):
    def __init__(self):
        self.__openAiTfIdfRepository = OpenAITfIdfRepositoryImpl()

        # 로컬 저장된 임베딩 데이터와 답변 리스트를 로드합니다.
        self.embeddingList, self.answerList = self.__load_local_data()

    def __load_local_data(self):
        # 임베딩과 답변을 로드하는 메서드입니다. 실제 구현은 로컬 파일 또는 DB에서 데이터 로드 로직에 따라 다를 수 있습니다.
        embeddingList = np.load(r'C:\TeamProject\SK-Networks-AI-1\TF\text-farmer-fastapi\app\assets\예방_Embeddings_test.pickle', allow_pickle=True)  # 예를 들면, NumPy 배열로 저장된 임베딩 데이터
        answerList = np.load(r'C:\TeamProject\SK-Networks-AI-1\TF\text-farmer-fastapi\app\assets\answers_8cols.pickle', allow_pickle=True)  # 답변 리스트
        return embeddingList, answerList

    async def letsTalk(self, userSendMessage):
        return await self.__openAiTfIdfRepository.generateText(userSendMessage)


    async def findBestResponse(self, userRequestText: str):
        # 사용자의 질문 임베딩
        userRequestEmbedding = self.__openAiTfIdfRepository.openAiBasedEmbedding(userRequestText)

        # FAISS 인덱스 생성 및 학습
        embeddingVectorDimension = len(self.embeddingList[0])
        faissIndex = self.__openAiTfIdfRepository.createL2FaissIndex(embeddingVectorDimension)
        embeddingMatrix = np.array(self.embeddingList).astype('float32')
        faissIndex.add(embeddingMatrix)

        # 유사도 분석
        indexList, distanceList = self.__openAiTfIdfRepository.similarityAnalysis(userRequestEmbedding, faissIndex)

        print(f"가장 유사한 답변 인덱스: {indexList}, 거리: {distanceList}")

        # 가장 유사한 답변 출력
        return OpenAITfIdfResponseForm.fromOpenAIPaperSimilarityAnalysis(indexList, distanceList, self.answerList)



