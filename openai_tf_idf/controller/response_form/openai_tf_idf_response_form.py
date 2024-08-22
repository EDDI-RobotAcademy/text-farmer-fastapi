from pydantic import BaseModel


class OpenAITfIdfResponseForm(BaseModel):
    similarity: float
    answer: str

    @classmethod
    def fromOpenAITfIdfSimilarityAnalysis(cls, indexList, distanceList, answerList):
        resultList = []

        for index in indexList:
            similarity = 1 - distanceList[index]
            resultList.append(cls(similarity=round(similarity, 4), answer=answerList[index]))

        return resultList
