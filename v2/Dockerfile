# Use the official Python 3.12 slim image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (needed for compiling some PyTorch/Ray C++ extensions)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 🚀 The official way to install uv in Docker
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy your project definition files
# The asterisk allows it to gracefully copy uv.lock if you have one, or skip it if you don't
COPY pyproject.toml uv.lock* ./

# Sync the dependencies (this creates a .venv inside the container)
# We use --no-install-project to just fetch the dependencies first for Docker caching
RUN uv sync --no-dev --no-install-project

# Now copy your actual poker code
COPY . .

# Run the project using `uv run`, which automatically uses the container's .venv
#CMD ["uv", "run", "main.py"]
CMD uv run ray start --head --dashboard-host=0.0.0.0 --port=6379 --ray-client-server-port=10001 && uv run python main.py