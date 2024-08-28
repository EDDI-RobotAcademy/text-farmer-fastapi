from abc import ABC, abstractmethod


class OpenAITfIdfRepository(ABC):
    @abstractmethod
    def similarityAnalysis(self, userDefinedReceiverFastAPIChannel):
        pass
