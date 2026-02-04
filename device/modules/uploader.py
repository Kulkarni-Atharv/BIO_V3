
import threading
import time
import requests
import json
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.config import API_BASE_URL, DEVICE_ID
from device.modules.database import LocalDatabase

logger = logging.getLogger("Uploader")
logging.basicConfig(level=logging.INFO)

class DataUploader:
    def __init__(self, db: LocalDatabase, interval=5):
        self.db = db
        self.interval = interval
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info("Uploader service started.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Uploader service stopped.")

    def _run_loop(self):
        while self.running:
            try:
                self._sync_data()
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
            time.sleep(self.interval)

    def _sync_data(self):
        # Fetch unsynced records
        records = self.db.get_unsynced_records()
        if not records:
            return

        payload = []
        record_ids = []
        
        for record in records:
            r_id, r_timestamp, r_device_id, r_name = record
            payload.append({
                "timestamp": r_timestamp,
                "device_id": r_device_id,
                "name": r_name
            })
            record_ids.append(r_id)

        try:
            # Send to server
            response = requests.post(f"{API_BASE_URL}/attendance", json=payload, timeout=5)
            if response.status_code == 200:
                self.db.mark_as_synced(record_ids)
                logger.info(f"Synced {len(records)} records.")
            else:
                logger.warning(f"Failed to sync. Server returned: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Network unavailable: {e}")
