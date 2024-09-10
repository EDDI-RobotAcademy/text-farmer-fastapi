import pandas as pd

# CSV 파일 경로 설정
input_csv_file = r'/proj/text-farmer-fastapi/app/assets/answers_8cols.csv'
output_csv_file = r'/proj/text-farmer-fastapi/app/assets/진단_original_answers.csv'

# CSV 파일을 DataFrame으로 로드
df = pd.read_csv(input_csv_file)

# 필터링 조건에 맞는 데이터 추출
filtered_df = df[(df['disease_category'] == '응급질환') &
                 (df['intention'] == '진단')]

# 필터링된 데이터를 새로운 CSV 파일로 저장
filtered_df.to_csv(output_csv_file, index=False)

print(f"Filtered data has been saved to {output_csv_file}")