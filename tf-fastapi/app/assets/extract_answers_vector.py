import pickle, time
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def answerVectorization(answerlist):
    print("백터화 시작")
    countVectorizer = CountVectorizer()
    countMatrix = countVectorizer.fit_transform(answerlist)
    print("백터화 종료")

    return countVectorizer, countMatrix, answerlist

def saveAsPickle(filename, objects):
    with open(f"{filename}.pickle", "wb") as file:
        for obj in objects:
            pickle.dump(obj, file)


if __name__ == '__main__':
    answerDf = pd.read_pickle("answers.pickle")

    start_time = time.time()

    introAnswerList = answerDf['answer_intro'].tolist()
    saveAsPickle(
        "introAnswerVectorization",
        answerVectorization(introAnswerList))

    end_time = time.time()
    print(f"introAnswerVectorization 소요시간 : {end_time - start_time} 초", end="\n")


    start_time = time.time()

    totalAnswerList = (answerDf['answer_intro'] + " " +
                       answerDf['answer_body'] + " " +
                       answerDf['answer_conclusion']).tolist()
    saveAsPickle(
        "totalAnswerVectorization",
        answerVectorization(totalAnswerList))

    end_time = time.time()
    print(f"totalAnswerVectorization 소요시간 : {end_time - start_time} 초")
