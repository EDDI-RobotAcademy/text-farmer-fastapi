import os
import pickle
import time
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import pandas as pd


class BioBERTEmbedder:
    def __init__(self):
        # Biobert 모델과 토크나이저 초기화
        self.tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        self.model = BertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        self.model.eval()

        # GPU 사용 가능 시 모델을 GPU로 이동
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)

    def embed_text(self, text):
        # 입력 텍스트를 토큰화하고 모델에 입력
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(
            self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # [CLS] 토큰의 임베딩을 반환
        return outputs.last_hidden_state[:, 0, :].squeeze()

    def process_batch(self, text_batch):
        # 배치 단위로 임베딩 계산
        embeddings = []
        for text in text_batch:
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        return torch.stack(embeddings)  # 임베딩을 하나의 텐서로 결합


def save_embeddings_to_pickle(filename, embeddings):
    with open(f"{filename}.pickle", "wb") as file:
        pickle.dump(embeddings.cpu(), file)  # GPU 텐서를 CPU로 이동 후 저장


if __name__ == '__main__':
    answerDf = pd.read_pickle("answers_8cols.pickle")

    # valid_categories = ["감염성질환", "성형미용 및 재건", "여성질환", "응급질환", "피부질환"]
    valid_categories = ["성형미용 및 재건", "피부질환"]
    filtered_df = answerDf[(answerDf['disease_category'].isin(valid_categories)) &
                           (answerDf['intention'] == "진단")]


    # 데이터 준비
    totalAnswerList = (filtered_df['answer_intro'] + " " +
                       filtered_df['answer_body'] + " " +
                       filtered_df['answer_conclusion']).tolist()

    # Biobert 임베딩 준비
    embedder = BioBERTEmbedder()

    start_time = time.time()

    # 배치 처리 (메모리 문제를 피하기 위해 적절한 배치 크기 설정 필요)
    batch_size = 32
    embeddings_list = []
    for i in tqdm(range(0, len(totalAnswerList), batch_size)):
        batch = totalAnswerList[i:i + batch_size]
        batch_embeddings = embedder.process_batch(batch)
        embeddings_list.append(batch_embeddings)

    # 전체 임베딩 텐서를 하나로 결합
    all_embeddings = torch.cat(embeddings_list, dim=0)

    # Pickle 파일로 저장
    save_embeddings_to_pickle("totalAnswerEmbeddings", all_embeddings)

    end_time = time.time()
    print(f"totalAnswerEmbeddings 소요시간 : {end_time - start_time} 초")
