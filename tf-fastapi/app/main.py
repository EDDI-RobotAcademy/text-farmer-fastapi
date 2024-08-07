from fastapi import FastAPI

from tf_idf_bow.controller.tf_idf_bow_controller import tfIdfBowRouter

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

app.include_router(tfIdfBowRouter)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=33333)