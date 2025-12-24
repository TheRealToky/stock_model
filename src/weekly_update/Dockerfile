FROM python:3.14-alpine3.23

WORKDIR /app
COPY . /app

# Install system packages and Python deps
RUN apk update && \
    apk add --no-cache \
        bash \
        postgresql-client \
        curl \
        tzdata && \
    pip install --no-cache-dir -r requirements.txt

# Latest releases available at https://github.com/aptible/supercronic/releases
ENV SUPERCRONIC_URL=https://github.com/aptible/supercronic/releases/download/v0.2.40/supercronic-linux-amd64 \
    SUPERCRONIC_SHA1SUM=8ab9683df4b8287ba1e09923101d6ebf532b6eba \
    SUPERCRONIC=supercronic-linux-amd64

RUN curl -fsSLO "$SUPERCRONIC_URL" \
 && echo "${SUPERCRONIC_SHA1SUM}  ${SUPERCRONIC}" | sha1sum -c - \
 && chmod +x "$SUPERCRONIC" \
 && mv "$SUPERCRONIC" "/usr/local/bin/${SUPERCRONIC}" \
 && ln -s "/usr/local/bin/${SUPERCRONIC}" /usr/local/bin/supercronic

COPY pipeline_cron /etc/crontabs/root
RUN chmod 0644 /etc/crontabs/root

RUN chmod +x fetch_weekly.py

# Run supercronic in foreground with crontab
CMD ["supercronic", "/etc/crontabs/root"]
