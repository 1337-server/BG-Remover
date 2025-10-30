FROM python:3.12-slim

ARG RUNTIME=cli
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/models \
    RUNTIME=${RUNTIME}

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . ./

RUN mkdir -p ${MODEL_DIR}

EXPOSE 8080

RUN cat <<'EOF' > /app/entrypoint.sh
#!/bin/sh
# Entrypoint dispatches runtime-specific commands based on the RUNTIME environment variable.
set -e

case "${RUNTIME:-cli}" in
    flask)
        exec gunicorn "runtimes.flask.app:create_app()" --bind 0.0.0.0:8080
        ;;
    gui)
        exec python -m runtimes.gui.bg_remover_gui
        ;;
    *)
        exec python -m runtimes.cli.bgr_cli --help
        ;;
esac
EOF

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
