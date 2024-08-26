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
        try:
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response['data'][0]['embedding']
        except Exception as e:
            print(f"OpenAI API 요청 실패: {e}")
            return None

def embed_and_save_answers():
    # OpenAI 임베더 초기화
    embedder = OpenAIEmbedder(openaiApiKey)

    # 데이터 로드
    df = pd.read_pickle('answers_8cols.pickle')

    # 필터링할 범주와 의도
    categories = ["감염성질환"]
    intentions = ["예방"]

    # '홍역'인 데이터만 필터링
    df = df[df['disease_name_kor'] == '홍역']

    # 원본 텍스트 데이터를 저장할 사전 초기화
    original_text_dict = {intention: [] for intention in intentions}

    # 데이터 필터링 및 원본 텍스트 저장
    for intention in intentions:
        filtered_df = df[(df['disease_category'].isin(categories)) & (df['intention'] == intention)]

        # 답변 텍스트 결합
        totalAnswerList = (filtered_df['answer_intro'] + " " +
                           filtered_df['answer_body'] + " " +
                           filtered_df['answer_conclusion']).tolist()

        # 원본 텍스트 저장
        original_text_dict[intention] = totalAnswerList

    # 원본 텍스트를 pickle 파일로 저장
    with open("예방_Original_Texts.pickle", "wb") as file:
        pickle.dump(original_text_dict, file)

    # # 데이터 임베딩 및 저장
    # embeddings_dict = {intention: [] for intention in intentions}
    #
    # for intention in intentions:
    #     for text in tqdm(original_text_dict[intention], desc=f"Embedding for {intention}"):
    #         embedding = embedder.embed_text(text)
    #         embeddings_dict[intention].append(embedding)
    #
    # for intention, embeddings in embeddings_dict.items():
    #     with open(f"{intention}_Embeddings_test.pickle", "wb") as file:
    #         pickle.dump(embeddings, file)

if __name__ == '__main__':
    embed_and_save_answers()
