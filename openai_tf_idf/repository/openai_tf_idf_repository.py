from abc import ABC, abstractmethod


class OpenAITfIdfRepository(ABC):

    @abstractmethod
    def generateText(self, userSendMessage):
        pass

    @abstractmethod
    def similarityAnalysis(self, userRequestPaperTitle, faissIndex):
        pass

    @abstractmethod
    def openAiBasedEmbedding(self, paperTitleList):
        pass


    @abstractmethod
    def createL2FaissIndex(self, embeddingVectorDimension):
        pass