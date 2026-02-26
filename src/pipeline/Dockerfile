FROM python:3.14-alpine3.23

WORKDIR /app
COPY . /app

RUN apk update && apk upgrade \
    && apk add --no-cache bash postgresql-client \
    && pip install --no-cache-dir -r requirements.txt

RUN chmod +x run_pipeline.sh

CMD ["bash", "run_pipeline.sh"]
