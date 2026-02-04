
import uvicorn
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.config import SERVER_PORT

if __name__ == "__main__":
    print(f"Starting Server on port {SERVER_PORT}...")
    uvicorn.run("server.api:app", host="0.0.0.0", port=SERVER_PORT, reload=True)
