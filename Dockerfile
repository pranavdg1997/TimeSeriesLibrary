FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install base Python dependencies
RUN pip install --upgrade pip

# Install package in development mode with CPU dependencies
RUN pip install -e ".[dev]"

# For GPU version, use Dockerfile.gpu instead

# Default command
CMD ["bash"]
