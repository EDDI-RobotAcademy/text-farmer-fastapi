FROM arm64v8/python:3.12

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 33333


CMD ["sh", "-c", "app.main:app"]
