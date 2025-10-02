 
#!/bin/bash

APP_NAME="churn-api"
DOCKER_IMAGE="churn-api:latest"

echo "[INFO] Pulling latest code..."
cd ~/churn-api || exit 1
git pull origin main

echo "[INFO] Stopping and removing old container..."
docker stop $APP_NAME || true
docker rm $APP_NAME || true

echo "[INFO] Building Docker image..."
docker build -t $DOCKER_IMAGE .

echo "[INFO] Running new container..."
docker run -d --name $APP_NAME -p 80:8000 $DOCKER_IMAGE

echo "[SUCCESS] App deployed and running on port 80."
