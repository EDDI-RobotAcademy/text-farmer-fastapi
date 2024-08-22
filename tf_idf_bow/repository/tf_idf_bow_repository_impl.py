import json
import queue

from tf_idf_bow.repository.tf_idf_bow_repository import TfIdfBowRepository


class TfIdfBowRepositoryImpl(TfIdfBowRepository):
    def getAnswer(self, userDefinedReceiverFastAPIChannel):
        print(f"TfIdfBowRepositoryImpl getAnswer()")

        try:
            receivedResponseFromSocketClient = userDefinedReceiverFastAPIChannel.get(False)
            return json.loads(receivedResponseFromSocketClient)

        except queue.Empty:
            return "아직 데이터를 처리 중이거나 요청한 데이터가 없습니다."
