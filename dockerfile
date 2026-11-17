
# 1. Official lightweight Python image
FROM python:3.11-slim

# 2. Working directory
WORKDIR /app

# 3. Copy dependencies
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy project files
COPY . .

# Copy GitHub-tracked deployment model
COPY src/serving/model/deployment_model/ /app/model/
# 8. Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 9. Expose Streamlit port
EXPOSE 8501

# 10. Start Streamlit
CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
