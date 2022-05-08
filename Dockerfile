# syntax=docker/dockerfile:1

FROM python:3.9.12-buster

WORKDIR app/

RUN apt update && apt install -y ffmpeg

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY gensim/ ./gensim
copy main.py .

CMD ["python3", "main.py"]


