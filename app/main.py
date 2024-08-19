from fastapi import FastAPI
from starlette.websockets import WebSocket, WebSocketDisconnect

from tf_idf_bow.controller.tf_idf_bow_controller import tfIdfBowRouter

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

app.include_router(tfIdfBowRouter)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    app.state.connections.add(websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        app.state.connections.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.0.55", port=33333)