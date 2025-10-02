#!/bin/bash

echo "🔍 Checking file permissions..."

important_files=(
  "main.py"
  "inference.py"
  "churnexplainer.py"
  "requirements.txt"
  "Dockerfile"
  "deploy.sh"
  "start.sh"
  ".github/workflows/deploy.yml"
  "models/telco_linear/telco_linear.pkl"
)

for file in "${important_files[@]}"; do
  if [ -e "$file" ]; then
    perms=$(stat -c "%a %n" "$file")
    echo "✅ $perms"
  else
    echo "❌ $file not found!"
  fi
done
