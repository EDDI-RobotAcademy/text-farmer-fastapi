import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


answerDf = pd.read_pickle("answers.pickle")

introAnswerList = answerDf['answer_intro'].tolist()
totalAnswerList = (answerDf['answer_intro'] + " " +
                   answerDf['answer_body'] + " " +
                   answerDf['answer_conclusion']).tolist()

def answerVectorization(answerlist):
    countVectorizer = CountVectorizer()
    countMatrix = countVectorizer.fit_transform(answerlist)

    return countVectorizer, countMatrix, answerlist

def saveAsPickle(filename, objects):
    for obj in objects:
        with open(f"{filename}.pickle", "wb") as file:
            pickle.dump(obj, file)

saveAsPickle(
    "introAnswerVectorization",
    answerVectorization(introAnswerList[:2]))

# saveAsPickle(
#     "totalAnswerVectorization",
#     answerVectorization(totalAnswerList))

