from abc import ABC, abstractmethod

class TfIdfBowRepository(ABC):
    @abstractmethod
    def getAnswer(self, userDefinedReceiverFastAPIChannel):
        pass
