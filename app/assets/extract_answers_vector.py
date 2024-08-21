import os
import pickle, time
import pandas as pd
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def tokenize_and_stem(text):
    okt = Okt()     # Open Korean Text: 오픈 소스 한국어 분석기
    morphs = okt.pos(text)  # 각 단어마다 품사 부착 [('word', 'pos')] 구조
    stems = []
    for word, pos in morphs:
        # 품사가 '동사', '형용사' 라면 형태소 단위로 나눠서 어간을 추출한 후 저장
        if pos in ['Verb', 'Adjective']:
            stems.append(okt.morphs(word, stem=True)[0])
        # 다른 품사라면 그대로 저장
        else:
            stems.append(word)
    return ' '.join(stems)


def preprocess_texts(text_list):
    print("형태소 분석 및 어간 추출 시작")

    processed_texts = [tokenize_and_stem(text) for text in tqdm(text_list)]
    print("형태소 분석 및 어간 추출 종료")
    return processed_texts


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
    answerDf = pd.read_pickle("answers_8cols.pickle")

    # # intro 부분만 vectorize
    # start_time = time.time()

    # introAnswerList = answerDf['answer_intro'].tolist()
    # saveAsPickle(
    #     "introAnswerVectorization",
    #     answerVectorization(introAnswerList))
    #
    # end_time = time.time()
    # print(f"introAnswerVectorization 소요시간 : {end_time - start_time} 초", end="\n")


    # # intro, body, conclusion 모두 vectorize
    # start_time = time.time()
    #
    # totalAnswerList = (answerDf['answer_intro'] + " " +
    #                    answerDf['answer_body'] + " " +
    #                    answerDf['answer_conclusion']).tolist()
    # saveAsPickle(
    #     "totalAnswerVectorization",
    #     answerVectorization(totalAnswerList))
    #
    # end_time = time.time()
    # print(f"totalAnswerVectorization 소요시간 : {end_time - start_time} 초")

    # intro, body, conclusion 어간 추출 후 vectorize
    start_time = time.time()

    totalAnswerList = (answerDf['answer_intro'] + " " +
                       answerDf['answer_body'] + " " +
                       answerDf['answer_conclusion']).tolist()

    # 형태소 분석 및 어간 추출 수행
    totalAnswerStemList = preprocess_texts(totalAnswerList)

    saveAsPickle(
        "totalAnswerStemVectorization",
        answerVectorization(totalAnswerStemList))

    end_time = time.time()
    print(f"totalAnswerStemVectorization 소요시간 : {end_time - start_time} 초")
