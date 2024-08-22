import numpy as np
from openai_tf_idf.controller.response_form.openai_paper_similarity_analysis_response_form import \
    OpenAIPaperSimilarityAnalysisResponseForm
from openai_tf_idf.repository.openai_tf_idf_repository_impl import OpenAITfIdfRepositoryImpl
from openai_tf_idf.service.openai_tf_idf_service import OpenAITfIdfService


class OpenAITfIdfServiceImpl(OpenAITfIdfService):
    def __init__(self):
        self.__openAiTfIdfRepository = OpenAITfIdfRepositoryImpl()

    async def letsTalk(self, userSendMessage):
        return await self.__openAiTfIdfRepository.generateText(userSendMessage)

    async def textSimilarityAnalysis(self, paperTitleList, userRequestPaperTitle):
        embeddingList = [
            self.__openAiBasicRepository.openAiBasedEmbedding(paperTitle)
            for paperTitle in paperTitleList]

        embeddingVectorDimension = len(embeddingList[0])
        faissIndex = self.__openAiBasicRepository.createL2FaissIndex(embeddingVectorDimension)
        embeddingMatrix = np.array(embeddingList).astype('float32')
        faissIndex.add(embeddingMatrix)

        indexList, distanceList = (
            self.__openAiBasicRepository.similarityAnalysis(userRequestPaperTitle, faissIndex))

        print(f"indexList: {indexList}, distanceList: {distanceList}")

        return OpenAIPaperSimilarityAnalysisResponseForm.fromOpenAIPaperSimilarityAnalysis(
            indexList, distanceList, paperTitleList
        )



