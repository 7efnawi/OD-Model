from fastapi import FastAPI
import uvicorn
import os
import threading
import time
import subprocess

app = FastAPI()

# Global flag to track main app status
main_app_started = False

@app.get("/")
def root():
    return {"message": "Health check server running"}

@app.get("/health")
def health():
    # Always return OK for health checks
    return {"status": "ok"}

def start_main_app():
    global main_app_started
    
    # Wait a bit to let this health check server start completely
    time.sleep(5)
    
    # Get the PORT from environment or use 8001 as a fallback
    port = os.environ.get("PORT", "8000")
    
    print(f"Starting main application...")
    
    # Start the main app with a different port
    main_port = str(int(port) + 1)
    os.environ["MAIN_PORT"] = main_port
    
    # Indicate that the main app is starting
    main_app_started = True
    
    # Use subprocess to start the main app
    subprocess.run(["python", "main.py"])

if __name__ == "__main__":
    # Start a thread to run the main app
    thread = threading.Thread(target=start_main_app)
    thread.daemon = True
    thread.start()
    
    # Get the port from environment or default to 8000
    port = int(os.environ.get("PORT", "8000"))
    print(f"Starting health check server on port {port}")
    
    # Start health check server
    uvicorn.run(app, host="0.0.0.0", port=port) 