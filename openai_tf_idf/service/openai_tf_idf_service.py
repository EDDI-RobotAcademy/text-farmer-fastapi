from abc import ABC, abstractmethod


class OpenAITfIdfService(ABC):

    @abstractmethod
    def letsTalk(self, userSendMessage):
        pass

    @abstractmethod
    def findBestResponse(self, userRequestText, embeddingList, answerList):
        pass