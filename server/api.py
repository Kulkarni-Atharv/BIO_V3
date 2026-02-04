
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List
import uvicorn
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from server.database import ServerDatabase
from server.mqtt_client import MQTTPublisher

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

app = FastAPI(title="Smart Attendance Server")

# Initialize modules
# NOTE: Update connection string for production SQL Server
db_connection_string = None 
db = ServerDatabase(connection_string=db_connection_string)
mqtt_publisher = MQTTPublisher()

class AttendanceRecord(BaseModel):
    timestamp: float
    device_id: str
    name: str

@app.post("/api/attendance")
async def receive_attendance(records: List[AttendanceRecord], background_tasks: BackgroundTasks):
    logger.info(f"Received {len(records)} records.")
    
    for record in records:
        # Save to DB
        success = db.insert_attendance(record.device_id, record.name, record.timestamp)
        if not success:
            logger.error("Failed to save record to DB.")
        
        # Publish to MQTT (using background task to not block response)
        background_tasks.add_task(mqtt_publisher.publish_attendance, record.device_id, record.name, record.timestamp)
    
    return {"status": "success", "processed": len(records)}

@app.get("/health")
def health_check():
    return {"status": "online"}
