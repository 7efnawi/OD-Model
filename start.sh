#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

# Use PORT environment variable or default to 8000
PORT="${PORT:-8000}"

# Print the port for logging purposes
echo "Starting server on port $PORT"

# Execute uvicorn using the determined port
exec uvicorn main:app --host 0.0.0.0 --port "$PORT" 