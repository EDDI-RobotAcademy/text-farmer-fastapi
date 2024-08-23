import pandas as pd

# 데이터 로드
df = pd.read_pickle('예방_Embeddings_test.pickle')

# 결측값 확인
print(df)
