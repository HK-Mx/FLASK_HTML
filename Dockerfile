FROM python:3.10.18-slim

RUN mkdir FLASK_AI_DOCKER
WORKDIR /FLASK_AI_DOCKER

COPY app.py ./
COPY requirements.txt ./
COPY model.pkl ./
COPY .env ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080","app:app"]
