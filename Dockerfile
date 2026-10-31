# MetaRisk XAI — Docker deployment
# Build:  docker build -t metarisk-xai .
# Run:    docker run -p 8501:8501 --env-file .env metarisk-xai

FROM python:3.11-slim

# System dependencies for LightGBM / ReportLab
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
