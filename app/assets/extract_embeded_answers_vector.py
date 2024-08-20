import os
import pickle
import time
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm


class BioBERTEmbedder:
    def __init__(self):
        # Biobert 모델과 토크나이저 초기화
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        self.model = BertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1").to(self.device)
        self.model.eval()

    def embed_text(self, text):
        # 입력 텍스트를 토큰화하고 모델에 입력
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
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
        return torch.stack(embeddings)


def save_embeddings_to_pickle(filename, embeddings, texts):
    with open(f"{filename}.pickle", "wb") as file:
        pickle.dump((embeddings, texts), file)


def load_embeddings_from_pickle(filename):
    with open(f"{filename}.pickle", "rb") as file:
        return pickle.load(file)


if __name__ == '__main__':
    answerDf = pd.read_pickle("answers_8cols.pickle")

    # 데이터 준비
    totalAnswerList = (answerDf['answer_intro'] + " " +
                       answerDf['answer_body'] + " " +
                       answerDf['answer_conclusion']).tolist()

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
    all_embeddings = torch.cat(embeddings_list, dim=0).cpu().numpy()  # GPU에서 계산된 결과를 CPU로 이동 후 Numpy 배열로 변환

    # Pickle 파일로 저장
    save_embeddings_to_pickle("totalAnswerEmbeddings", all_embeddings, totalAnswerList)

    end_time = time.time()
    print(f"totalAnswerEmbeddings 소요시간 : {end_time - start_time} 초")
