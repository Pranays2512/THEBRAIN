FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    python3-dev \
    libomp-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python dependencies
RUN pip3 install --no-cache-dir pybind11 numpy torch fastapi uvicorn pydantic

# Copy project files
COPY . /app/

# Remove any existing build artifacts that might have been copied from the host
RUN rm -rf brain2/build

# Build the C++ extension
RUN mkdir -p brain2/build && \
    cd brain2/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# Set the working directory to brain2 where the python scripts are executed from
WORKDIR /app/brain2

# Default command to run the interactive session
CMD ["python3", "brain_session.py"]
