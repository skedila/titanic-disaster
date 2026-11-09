FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY titanic_model.py .

# The data folder is mounted at runtime (see README), not copied into the image
ENV DATA_DIR=/app/data

CMD ["python", "titanic_model.py"]
