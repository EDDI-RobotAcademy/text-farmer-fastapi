import os
import pickle

import numpy as np

from openai_tf_idf.controller.response_form.openai_tf_idf_response_form import OpenAITfIdfSimilarityAnalysisResponseForm
from openai_tf_idf.repository.openai_tf_idf_repository_impl import OpenAITfIdfRepositoryImpl
from openai_tf_idf.service.openai_tf_idf_service import OpenAITfIdfService


class OpenAITfIdfServiceImpl(OpenAITfIdfService):
    def __init__(self):
        self.__openAiTfIdfRepository = OpenAITfIdfRepositoryImpl()
        self.embedding_pickle_path = r"C:\TeamProject\SK-Networks-AI-1\TF\text-farmer-fastapi\app\assets\예방_Embeddings_test.pickle"
        self.original_data_path = r"C:\TeamProject\SK-Networks-AI-1\TF\text-farmer-fastapi\app\assets\answers_8cols.pickle"

        # 임베딩된 데이터 로드
        with open(self.embedding_pickle_path, "rb") as file:
            self.embedded_answer = pickle.load(file)

        # FAISS 인덱스 생성 및 임베딩 데이터 추가
        embeddingVectorDimension = len(self.embedded_answer[0])
        self.faissIndex = self.__openAiTfIdfRepository.createL2FaissIndex(embeddingVectorDimension)
        self.faissIndex.add(np.array(self.embedded_answer).astype('float32'))

        # 가장 유사한 답변의 인덱스를 통해 원본 데이터에서 답변 가져오기
        with open(self.original_data_path, "rb") as file:
            self.original_answers = pickle.load(file)

    async def textSimilarityAnalysis(self, userQuestion):
        indexList, distanceList = (
            self.__openAiTfIdfRepository.similarityAnalysis(userQuestion, self.faissIndex))

        print(f"indexList: {indexList}, distanceList: {distanceList}")



        best_answers = [self.original_answers[i] for i in indexList]

        return OpenAITfIdfSimilarityAnalysisResponseForm.fromOpenAITfIdfSimilarityAnalysis(
            indexList, distanceList, best_answers
        )




