FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY Model/requirements.txt /app/Model/requirements.txt
RUN pip install --no-cache-dir -r /app/Model/requirements.txt

COPY Model /app/Model

ENTRYPOINT ["python", "/app/Model/train.py"]

