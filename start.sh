#!/bin/bash

# Go to the app directory
cd ~/churn-api

# Set up Python environment (optional but good practice)
sudo apt update
sudo apt install -y python3-pip

# Install dependencies
pip3 install -r requirements.txt

# Start FastAPI with Uvicorn on port 8000
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > app.log 2>&1 &
