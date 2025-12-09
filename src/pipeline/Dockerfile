FROM python:3.14-alpine3.23

WORKDIR /app

# Install parallel and any system dependencies
RUN apk update && apk upgrade
RUN apk upgrade py3-pip

COPY requirements.txt .
RUN apk add --no-cache parallel bash postgresql-client \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x run_pipeline.sh

CMD ["bash", "run_pipeline.sh"]