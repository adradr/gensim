# syntax=docker/dockerfile:1

FROM python:3.9.12-buster

RUN apt update
RUN apt install -y ffmpeg
RUN apt install tzdata -y 
ENV TZ="Europe/Budapest"

WORKDIR app/

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY gensim/ ./gensim
COPY main.py .

CMD ["python3", "main.py"]