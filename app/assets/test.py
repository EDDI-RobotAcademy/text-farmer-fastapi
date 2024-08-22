import pandas as pd

# 데이터 로드
df = pd.read_pickle('answers_8cols.pickle')

# 결측값 확인
print("\nCheck for missing values in the columns:")
print(df[['answer_intro', 'answer_body', 'answer_conclusion']].isnull().sum())

# 결측값이 있는 경우, 이를 처리 (예: 빈 문자열로 대체)
df['answer_intro'] = df['answer_intro'].fillna('')
df['answer_body'] = df['answer_body'].fillna('')
df['answer_conclusion'] = df['answer_conclusion'].fillna('')