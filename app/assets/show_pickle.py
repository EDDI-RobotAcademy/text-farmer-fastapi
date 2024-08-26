import pickle

def load_and_show_embeddings(intention):
    # 저장된 피클 파일 로드
    try:
        with open(f"{intention}_Embeddings_test.pickle", "rb") as file:
            embeddings = pickle.load(file)
            return embeddings
    except FileNotFoundError:
        print(f"파일 {intention}_Embeddings_test.pickle을 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"파일을 읽는 중 오류 발생: {e}")
        return None

if __name__ == '__main__':
    # 의도별로 임베딩을 로드하여 리스트 출력
    intention = "예방"  # 원하는 의도에 맞게 수정 가능
    embeddings_list = load_and_show_embeddings(intention)

    if embeddings_list is not None:

        for i, embedding in enumerate(embeddings_list):
            print(f"Embedding {i+1}: {embedding}")
    print(f"{intention} 임베딩 리스트 (총 {len(embeddings_list)}개):")
