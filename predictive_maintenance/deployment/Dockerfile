FROM python:3.13.5-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN useradd -m -u 1000 user
USER user
WORKDIR $HOME/app

# Copy dependency requirements and leverage pip caching mechanisms
COPY --chown=user requirements.txt .
RUN pip3 install --no-cache-dir --user -r requirements.txt

# Mirror app execution workspace files into container layers
COPY --chown=user . $HOME/app

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
