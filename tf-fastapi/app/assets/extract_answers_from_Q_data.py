import os
import pandas as pd
import json, pickle
from tqdm import tqdm

def makeFilePath(*args):
    return "\\".join(args)

def extractJsonData(data, keys):
    extracted_values = []
    for key in keys:
        if '.' in key:
            nested_keys = key.split('.')
            temp_data = data
            try:
                for nk in nested_keys:
                    temp_data = temp_data[nk]
                extracted_values.append(temp_data)
            except KeyError:
                continue
        else:
            if key in data:
                extracted_values.append(data[key])
    return extracted_values


# ---------------- [추출 데이터 값 설정] ----------------------
ROOT_PATH = r"C:\Users\USER\Desktop"
DATA_PATH = r"120.초거대AI 사전학습용 헬스케어 질의응답 데이터\3.개방데이터\1.데이터"
DATA_TYPE = {"train": "Training", "validation": "Validation",
             "raw": "01.원천데이터", "labeled": "02.라벨링데이터",
             "Q": r"1.질문", "A": r"2.답변"}
FILE_PATH = makeFilePath(ROOT_PATH, DATA_PATH,
                       DATA_TYPE["train"], DATA_TYPE["labeled"], DATA_TYPE["A"])
DISEASE_TYPE = os.listdir(FILE_PATH)

DESIRED_DATA = ["answer.intro", "answer.body", "answer.conclusion",
                "disease_category", "disease_name.kor", "department", "num_of_words"]
DESIRED_COLS = ["answer_intro", "answer_body", "answer_conclusion",
                "disease_category", "disease_name_kor", "department", "num_of_words"]
# ------------------------------------------------------------


result = []
failLoadData = []


for div_1 in DISEASE_TYPE:
    for div_2 in tqdm((os.listdir(makeFilePath(FILE_PATH, div_1))), desc=div_1):
        for div_3 in os.listdir(makeFilePath(FILE_PATH, div_1, div_2)):
            for file_name in os.listdir(makeFilePath(FILE_PATH, div_1, div_2, div_3)):
                try:
                    with open(makeFilePath(FILE_PATH, div_1, div_2, div_3, file_name), 'r', encoding='UTF-8-sig') as jsonFile:
                        jsonData = json.load(jsonFile)
                        extracted_data = extractJsonData(jsonData, DESIRED_DATA)
                        if extracted_data:
                            result.append(extracted_data)

                except:
                    failLoadData.append(makeFilePath(FILE_PATH, div_1, div_2, div_3, file_name))



df = pd.DataFrame(result, columns=DESIRED_COLS)
df.to_pickle('answers.pickle')
df.to_csv('answers.csv', index=False)

with open("failed_data.pickle", "wb") as file:
    pickle.dump(failLoadData, file)