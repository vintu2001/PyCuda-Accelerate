FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    cmake \
    ninja-build \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /workspace

COPY . /workspace

RUN python -m pip install --upgrade pip && \
    python -m pip install -e .[dev] -v

CMD ["pytest", "tests/", "-v", "--tb=short"]
