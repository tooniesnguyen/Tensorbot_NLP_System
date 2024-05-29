FROM python:3.9

RUN pip install -r requirements.txt

WORKDIR /usr/app/src

COPY hello_world.py ./