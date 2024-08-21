import os
import pickle
import time
import torch

from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Okt

from tf_idf_bow.repository.tf_idf_bow_repository import TfIdfBowRepository


class TfIdfBowRepositoryImpl(TfIdfBowRepository):
    VECTORIZATION_FILE_PATH = os.path.join(
        os.getcwd(), "assets", "totalAnswerStemVectorization.pickle"
    )
    RAW_ANSWERS_FILE_PATH = os.path.join(
        os.getcwd(), "assets", "answers_8cols.pickle"
    )
    TOP_RANK_LIMIT = 3
    SIMILARITY_THRESHOLD = 0.1

    def __init__(self):
        self.okt = Okt()

    def tokenize_and_stem(self, text):
        morphs = self.okt.pos(text)
        stems = []
        for word, pos in morphs:
            if pos in ['Verb', 'Adjective']:
                stems.append(self.okt.morphs(word, stem=True)[0])
            else:
                stems.append(word)
        return ' '.join(stems)

    def findSimilarText(self, userQuestion):
        stime = time.time()

        # 형태소 분석 및 어간 추출을 사용자 질문에 적용
        processed_userQuestion = self.tokenize_and_stem(userQuestion)

        with open(self.VECTORIZATION_FILE_PATH, "rb") as pickleFile:
            countVectorizer = pickle.load(pickleFile)
            countMatrix = pickle.load(pickleFile)
            # answerList = pickle.load(pickleFile)

        with open(self.RAW_ANSWERS_FILE_PATH, "rb") as pickleFile:
            allAnswerData = pickle.load(pickleFile)

        # userQuestionVector = countVectorizer.transform([userQuestion])
        userQuestionVector = countVectorizer.transform([processed_userQuestion])
        cosineSimilarityList = cosine_similarity(userQuestionVector, countMatrix).flatten()
        similarIndexList = cosineSimilarityList.argsort()[-self.TOP_RANK_LIMIT:][::-1]
        similarAnswerList = [allAnswerData.iloc[index, :]
                             for index in similarIndexList
                             if cosineSimilarityList[index] >= self.SIMILARITY_THRESHOLD]
        etime = time.time()

        similarityValueList = [cosineSimilarityList[index]
                             for index in similarIndexList
                             if cosineSimilarityList[index] >= self.SIMILARITY_THRESHOLD]

        print(similarAnswerList)
        print(similarityValueList)
        print(f"{etime-stime} 초")

        return similarAnswerList