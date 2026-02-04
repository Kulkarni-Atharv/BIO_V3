
import sqlite3
import time
import os
import sys

# Get absolute path to BIO/ and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.config import DB_PATH

class LocalDatabase:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    device_id TEXT,
                    name TEXT,
                    synced BOOLEAN DEFAULT 0
                )
            ''')
            conn.commit()

    def add_record(self, device_id, name):
        timestamp = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO attendance_log (timestamp, device_id, name, synced)
                VALUES (?, ?, ?, 0)
            ''', (timestamp, device_id, name))
            conn.commit()
            return cursor.lastrowid

    def get_unsynced_records(self, limit=50):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, timestamp, device_id, name
                FROM attendance_log
                WHERE synced = 0
                LIMIT ?
            ''', (limit,))
            return cursor.fetchall()

    def mark_as_synced(self, record_ids):
        if not record_ids:
            return
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            placeholders = ', '.join(['?'] * len(record_ids))
            cursor.execute(f'''
                UPDATE attendance_log
                SET synced = 1
                WHERE id IN ({placeholders})
            ''', record_ids)
            conn.commit()
