from pydantic import BaseModel
from typing import List, Any

class OpenAITfIdfSimilarityAnalysisResponseForm(BaseModel):
    index_list: List[int]
    distance_list: List[float]
    best_answers: List[Any]

    @classmethod
    def fromOpenAITfIdfSimilarityAnalysis(cls, index_list, distance_list, best_answers):
        return cls(
            index_list=index_list.tolist(),
            distance_list=distance_list.tolist(),
            best_answers=best_answers
        )

    def __str__(self):
        return f"Indices: {self.index_list}\nDistances: {self.distance_list}\nBest Answers:\n{self.best_answers}"
