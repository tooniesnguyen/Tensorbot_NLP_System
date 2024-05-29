FROM python:3.9

EXPOSE 8008

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /usr/app/Tensorbot_NLP_System

COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:8008", "-k", "uvicorn.workers.UvicornWorker", "app:app"]