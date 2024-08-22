import os
import pickle
import time
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Okt

class BioBERTEmbedder:
    def __init__(self):
        # Biobert 모델과 토크나이저 초기화
        self.tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        self.model = BertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        self.model.eval()
        # GPU 사용 시 모델을 GPU로 이동
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # [CLS] 토큰의 임베딩을 반환
        return outputs.last_hidden_state[:, 0, :].cpu()  # CPU로 반환

    def process_batch(self, text_batch):
        embeddings = []
        for text in text_batch:
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        return torch.stack(embeddings)  # 임베딩을 하나의 텐서로 결합

class TfIdfBowRepositoryImpl:
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
        self.embedder = BioBERTEmbedder()

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

        # 사용자 질문을 BiobERT로 임베딩
        userQuestionEmbedding = self.embedder.embed_text(processed_userQuestion).unsqueeze(0)  # 배치 차원 추가

        # 답변 데이터 임베딩을 로드
        with open(r"C:\TeamProject\SK-Networks-AI-1\TF\text-farmer-fastapi\app\assets\filteredAnswersEmbeddings.pickle", "rb") as file:
            totalAnswerEmbeddings = pickle.load(file)

        # 답변 데이터 임베딩과 사용자 질문 임베딩 간의 코사인 유사도 계산
        cosineSimilarityList = cosine_similarity(userQuestionEmbedding.cpu(), totalAnswerEmbeddings.cpu()).flatten()

        similarIndexList = cosineSimilarityList.argsort()[-self.TOP_RANK_LIMIT:][::-1]
        import pandas as pd
        df = pd.read_pickle(self.RAW_ANSWERS_FILE_PATH)
        similarAnswerList = [df.iloc[index, :] for index in similarIndexList
                             if cosineSimilarityList[index] >= self.SIMILARITY_THRESHOLD]
        etime = time.time()

        similarityValueList = [cosineSimilarityList[index]
                             for index in similarIndexList
                             if cosineSimilarityList[index] >= self.SIMILARITY_THRESHOLD]

        print(similarAnswerList)
        print(similarityValueList)
        print(f"{etime - stime} 초")

        return similarAnswerList
