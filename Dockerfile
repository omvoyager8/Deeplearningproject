# Base image with Python
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 3000 (same as in your workflow)
EXPOSE 3000

# Start your BentoML service
CMD ["bentoml", "serve", "."]
