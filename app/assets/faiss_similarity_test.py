import os
import pickle
import openai
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# OpenAI API Key 설정
openaiApiKey = os.getenv('OPENAI_API_KEY')
if not openaiApiKey:
    raise ValueError('API Key가 준비되어 있지 않습니다!')

class OpenAIEmbedder:
    def __init__(self, openaiApiKey):
        # OpenAI API 초기화
        openai.api_key = openaiApiKey

    def embed_text(self, text):
        try:
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return np.array(response['data'][0]['embedding'], dtype=np.float32)
        except Exception as e:
            print(f"OpenAI API 요청 실패: {e}")
            return None

def load_embeddings(intention):
    with open(f"{intention}_Embeddings_test.pickle", "rb") as file:
        embeddings = pickle.load(file)
    return embeddings

def load_original_texts():
    with open("예방_Original_Texts.pickle", "rb") as file:
        original_text_dict = pickle.load(file)
    return original_text_dict

def build_faiss_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)  # L2 거리(유클리드 거리) 사용
    index.add(np.array(embeddings, dtype=np.float32))  # 벡터 추가
    return index

def find_most_similar_answer_faiss(question_embedding, faiss_index):
    distances, indices = faiss_index.search(np.array([question_embedding]), k=1)
    return indices[0][0], distances[0][0]

def get_original_answer_by_index(index, original_text_dict, intention):
    return original_text_dict[intention][index]

if __name__ == '__main__':
    intention = "예방"  # 원하는 의도를 설정합니다.

    # 1. 임베딩된 데이터와 원본 텍스트 로드
    embeddings_list = load_embeddings(intention)
    original_text_dict = load_original_texts()

    # 2. FAISS 인덱스 구축
    faiss_index = build_faiss_index(embeddings_list)

    # 3. 사용자 질문 임베딩 생성
    embedder = OpenAIEmbedder(openaiApiKey)
    user_question = "홍역 예방 방법은 무엇인가요?"  # 예시 질문
    question_embedding = embedder.embed_text(user_question)

    if question_embedding is not None:
        # 4. 유사도 분석하여 가장 유사한 답변의 인덱스 찾기
        index, distance = find_most_similar_answer_faiss(question_embedding, faiss_index)
        print(f"가장 유사한 답변의 인덱스: {index}, 거리: {distance}")

        # 5. 인덱스로 원본 답변 가져오기
        original_answer = get_original_answer_by_index(index, original_text_dict, intention)
        print("가장 유사한 원본 답변:")
        print(original_answer)
    else:
        print("질문 임베딩 생성 실패!")
