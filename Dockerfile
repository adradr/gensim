# syntax=docker/dockerfile:1

FROM python:3.9.12-buster

RUN apt update
RUN apt install -y ffmpeg
RUN apt install -y graphviz
RUN apt install -y tzdata
ENV TZ="Europe/Budapest"

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY gensim/ gensim
COPY main.py main.py
COPY .env .env

CMD ["python3", "main.py"]
