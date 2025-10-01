#!/bin/bash

echo "Stopping existing Uvicorn processes..."
pkill -f "uvicorn" || true

echo "Starting FastAPI app with Uvicorn..."
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > output.log 2>&1 &

echo "✅ Application started on port 8000"
