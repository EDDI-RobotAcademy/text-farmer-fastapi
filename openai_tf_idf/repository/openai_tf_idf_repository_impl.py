import json
import queue

from openai_tf_idf.repository.openai_tf_idf_repository import OpenAITfIdfRepository


class OpenAITfIdfRepositoryImpl(OpenAITfIdfRepository):

    def similarityAnalysis(self, userDefinedReceiverFastAPIChannel):
        print(f"OpenAITfIdfRepositoryImpl similarityAnalysis()")

        try:
            receivedResponseFromSocketClient = userDefinedReceiverFastAPIChannel.get(False)
            return json.loads(receivedResponseFromSocketClient)

        except queue.Empty:
            return "아직 데이터를 처리 중이거나 요청한 데이터가 없습니다."