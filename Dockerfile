# use python 3.10 slim image as base
FROM python:3.10-slim

# config working directory
WORKDIR /app

# install build-essential for any potential compilation needs (like numpy, scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy code to container
COPY . .

# expose port 8080
EXPOSE 8080

# run API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
