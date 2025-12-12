FROM python:3.10-slim

# ---- system deps ----
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---- working directory ----
WORKDIR /app

# ---- copy files ----
COPY . /app

# ---- python deps ----
RUN pip install --upgrade pip && \
    pip install \
        torch \
        transformers \
        datasets \
        accelerate \
        numpy \
        matplotlib \
        tqdm

# ---- default command ----
CMD ["bash"]
