# Use an official lightweight Python image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies (CPU versions to keep it lightweight)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir numpy matplotlib

# Copy the rest of the application code
COPY . .

# Create necessary directories
RUN mkdir -p models results

# Default command to keep container open or for instruction purposes
CMD ["python", "src/train.py", "--model", "lstm"]