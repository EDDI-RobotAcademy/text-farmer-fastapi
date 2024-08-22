import os
import pickle
import openai
import pandas as pd

from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

openaiApiKey = os.getenv('OPENAI_API_KEY')
if not openaiApiKey:
    raise ValueError('API Key가 준비되어 있지 않습니다!')

class OpenAIEmbedder:
    def __init__(self, openaiApiKey):
        # OpenAI API 초기화
        openai.api_key = openaiApiKey

    def embed_text(self, text):
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response['data'][0]['embedding']

def embed_and_save_answers():
    # OpenAI 임베더 초기화
    embedder = OpenAIEmbedder(openaiApiKey)

    # 데이터 로드
    df = pd.read_pickle('answers_8cols.pickle')

    # 필터링할 범주와 의도
    # categories = ["감염성질환", "성형미용 및 재건", "여성질환", "응급질환", "피부질환"]
    categories = ["감염성질환"]
    intentions = ["예방"]
    # intentions = ["예방", "원인", "정의", "증상", "진단", "치료"]

    # '홍역'인 데이터만 필터링
    df = df[df['disease_name_kor'] == '홍역']

    # intention별로 임베딩 저장을 위한 사전 초기화
    embeddings_dict = {intention: [] for intention in intentions}

    # 데이터 필터링 및 임베딩
    for intention in intentions:
        filtered_df = df[(df['disease_category'].isin(categories)) & (df['intention'] == intention)]

        # 답변 텍스트 결합
        totalAnswerList = (filtered_df['answer_intro'] + " " +
                           filtered_df['answer_body'] + " " +
                           filtered_df['answer_conclusion']).tolist()

        # 임베딩 수행
        for text in tqdm(totalAnswerList, desc=f"Embedding for {intention}"):
            embedding = embedder.embed_text(text)
            embeddings_dict[intention].append(embedding)

    # 각 의도별로 임베딩 저장
    for intention, embeddings in embeddings_dict.items():
        # 임베딩 텐서를 pickle 파일로 저장
        with open(f"{intention}_Embeddings_test.pickle", "wb") as file:
            pickle.dump(embeddings, file)

if __name__ == '__main__':
    embed_and_save_answers()
