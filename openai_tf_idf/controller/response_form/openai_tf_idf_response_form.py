from pydantic import BaseModel

class OpenAITfIdfSimilarityAnalysisResponseForm(BaseModel):
    similarity: float
    answer: dict

    @classmethod
    def fromOpenAITfIdfSimilarityAnalysis(cls, indexList, distanceList, answerList):
        resultList = []

        for index, faissIndex in enumerate(indexList):
            similarity = 1 - distanceList[index]
            resultList.append(cls(similarity=round(similarity, 4), answer=answerList[faissIndex]))

        return resultList
