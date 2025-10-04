# Multi-stage Dockerfile for both UI and API
FROM node:18-alpine AS ui-builder

# Build React UI
WORKDIR /app/ui
COPY ui/package*.json ./
COPY ui/vite.config.ts ./
COPY ui/tailwind.config.js ./
COPY ui/postcss.config.js ./
COPY ui/tsconfig*.json ./
RUN npm ci

COPY ui/src/ ./src/
COPY ui/index.html ./
COPY ui/vite-env.d.ts ./
COPY ui/vite.config.prod.ts ./
RUN npx vite build --config vite.config.prod.ts

# Python API stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (Docker-friendly, no Tesseract)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    nginx \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.docker.txt ./requirements.txt
RUN pip install --no-cache-dir --force-reinstall -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY run.py .

# Copy built UI from previous stage
COPY --from=ui-builder /app/ui/dist ./static

# Create data directories
RUN mkdir -p /app/data/documents /app/data/vectordb /app/logs

# Create nginx configuration for serving UI and proxying API
RUN echo 'server { \
    listen 80; \
    server_name localhost; \
    \
    # Serve React UI \
    location / { \
        root /app/static; \
        try_files $uri $uri/ /index.html; \
    } \
    \
    # Proxy API requests \
    location /api/ { \
        proxy_pass http://127.0.0.1:8000/api/; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \
        proxy_set_header X-Forwarded-Proto $scheme; \
    } \
    \
    # Proxy health check \
    location /health { \
        proxy_pass http://127.0.0.1:8000/health; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
    } \
    \
    # Proxy docs \
    location /docs { \
        proxy_pass http://127.0.0.1:8000/docs; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
    } \
    \
    location /redoc { \
        proxy_pass http://127.0.0.1:8000/redoc; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
    } \
}' > /etc/nginx/sites-available/default

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 80 8000

# Create startup script
RUN echo '#!/bin/bash\n\
# Start nginx in background\n\
nginx -g "daemon off;" &\n\
\n\
# Start Python API\n\
python run.py\n\
' > /app/start.sh && chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

# Run both services
CMD ["/app/start.sh"]