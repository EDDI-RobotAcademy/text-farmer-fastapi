import pandas as pd
import os
import json
from tqdm import tqdm


# ---------------- [추출 데이터 값 설정] ----------------------
ROOT_PATH = r"C:\Users\USER\Desktop"                 # 데이터를 다운받은 디렉토리 주소
DATA_DETAIL_PATH = {"DATA_TYPE_1": "Training",       # ["Training", "Validation"]
                    "DATA_TYPE_2": "02.라벨링데이터",  # ["01.원천데이터", "02.라벨링데이터"]
                    "DATA_TYPE_3": "2.답변"}          # ["1.질문", "2.답변"]
DISEASE_TYPE = ""
# DISEASE_TYPE = ["감염성질환", "눈질환"]               # 특정 질병분류로 추출 원할 시 주석 해제 후, 원하는 값 입력
# ------------------------------------------------------------
DATA_PATH = r"120.초거대AI 사전학습용 헬스케어 질의응답 데이터\3.개방데이터\1.데이터"
FILE_PATH = os.path.join(ROOT_PATH, DATA_PATH,
                         DATA_DETAIL_PATH["DATA_TYPE_1"],
                         DATA_DETAIL_PATH["DATA_TYPE_2"],
                         DATA_DETAIL_PATH["DATA_TYPE_3"])

DESIRED_DATA = ["disease_category", "disease_name.kor", "department", "intention",
                "answer.intro", "answer.body", "answer.conclusion", "num_of_words"]
DESIRED_COLS = ["disease_category", "disease_name_kor", "department", "intention",
                "answer_intro", "answer_body", "answer_conclusion", "num_of_words"]


def extract_data_from_json(file_path):
    try:
        with open(file_path, 'r', encoding='UTF-8-sig') as file:
            data = json.load(file)

        extracted_value_list = []
        for key in DESIRED_DATA:
            if '.' in key:
                key1, key2 = key.split(".")
                extracted_value_list.append(data[key1][key2])

            else:
                extracted_value_list.append(data[key])

        return extracted_value_list

    except Exception as e:
        print(f"Error-extract_data_from_json() : {e}")
        return None, file_path


def get_json_files(file_path):
    json_files = []

    for root, _, files in os.walk(file_path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))

    return json_files


def gather_json_files():
    if DISEASE_TYPE:
        json_files = []

        for dir in DISEASE_TYPE:
            file_path = os.path.join(FILE_PATH, dir)
            json_files += get_json_files(file_path)

        return json_files

    else:
        return get_json_files(FILE_PATH)


def process_without_threadpool():
    results = []
    for file_path in tqdm(json_files):
        result = extract_data_from_json(file_path)
        results.append(result)

    return results


if __name__ == '__main__':
    json_files = gather_json_files()
    answer_list = process_without_threadpool()

    df = pd.DataFrame(answer_list, columns=DESIRED_COLS)
    df.to_pickle('answers_8cols.pickle')
    df.to_csv('answers_8cols.csv', index=False)
