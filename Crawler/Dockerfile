FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
LABEL maintainer="disi-unibo-nlp"

# Zero interaction (default answers to all questions)
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /

# Install general-purpose dependencies
RUN apt-get update -y && \
    apt-get install -y curl \
    git \
    bash \
    nano \
    wget \
    python3.9 \
    python3-pip && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*


# Copy the requirements file into the container at /workspace
COPY requirements.txt .

RUN pip3 install --upgrade setuptools==69.5.1

# Install dependencies from requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# RUN pip3 install flash_attn==2.5.8 --no-build-isolation

# Milvus
# EXPOSE 19530

# Back to default frontend
ENV DEBIAN_FRONTEND=dialog