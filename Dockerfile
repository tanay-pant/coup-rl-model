FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install git and other system dependencies if Ray needs them
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
# Add web dependencies missing from original requirements
RUN pip install --no-cache-dir -r requirements.txt fastapi "uvicorn[standard]" websockets

# Copy the entire project
COPY . .

# Create a non-root user (Hugging Face Spaces requirement for Docker)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user . $HOME/app

EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
