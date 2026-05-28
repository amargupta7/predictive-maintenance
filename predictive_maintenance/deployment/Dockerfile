FROM python:3.13-slim

# Set the initial working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security compliance
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Adjust working directory to user space
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip3 install --no-cache-dir --user -r requirements.txt

# Copy the rest of your application code with correct ownership
COPY --chown=user . $HOME/app

# Expose Streamlit's default port
EXPOSE 8501

# Run your application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
