from fastapi import FastAPI
from starlette.websockets import WebSocket, WebSocketDisconnect

from user_defined_initializer.init import UserDefinedInitializer
from tf_idf_bow.controller.tf_idf_bow_controller import tfIdfBowRouter

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'template'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'template', 'include', 'socket_server'))

from template.deep_learning.controller.deep_learning_controller import deepLearningRouter
from template.dice.controller.dice_controller import diceResultRouter
from template.include.socket_server.initializer.init_domain import DomainInitializer
from template.system_initializer.init import SystemInitializer
from template.task_manager.manager import TaskManager

DomainInitializer.initEachDomain()
SystemInitializer.initSystemDomain()
UserDefinedInitializer.initUserDefinedDomain()

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

app.include_router(tfIdfBowRouter)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    app.state.connections.add(websocket)

app.include_router(deepLearningRouter)
app.include_router(diceResultRouter)

app.include_router(tfIdfBowRouter)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.0.55", port=33333)