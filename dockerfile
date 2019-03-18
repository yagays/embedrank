FROM python:3.6

RUN apt-get update && apt-get install -y \
    libmecab-dev \
    mecab-ipadic-utf8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . /app
WORKDIR /app/

RUN pip install -U pip
RUN pip install -r requirements.txt

WORKDIR /app/web/

ENTRYPOINT ["python", "api.py"]
