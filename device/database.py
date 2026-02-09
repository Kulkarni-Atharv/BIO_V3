import mysql.connector
import time
import os
import sys
import logging

# Get absolute path to BIO/ and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.config import MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB, MYSQL_PORT

logger = logging.getLogger("Database")

class LocalDatabase:
    def __init__(self):
        self._init_db()

    def _get_connection(self):
        try:
            return mysql.connector.connect(
                host=MYSQL_HOST,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DB,
                port=MYSQL_PORT
            )
        except mysql.connector.Error as err:
            logger.error(f"MySQL Connection Error: {err}")
            return None

    def _init_db(self):
        # Database should correspond to the one created via setup_mysql.sql
        # We can check connection here
        conn = self._get_connection()
        if conn:
            logger.info("Connected to MySQL Database successfully.")
            conn.close()
        else:
            logger.error("Failed to connect to MySQL Database. Check config and network.")

    def add_record(self, device_id, name):
        timestamp = time.time()
        conn = self._get_connection()
        if not conn:
            return None
        
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO attendance_log (timestamp, device_id, name, synced)
                VALUES (%s, %s, %s, 0)
            ''', (timestamp, device_id, name))
            conn.commit()
            return cursor.lastrowid
        except mysql.connector.Error as err:
            logger.error(f"Failed to add record: {err}")
            return None
        finally:
            if conn: conn.close()

    def get_unsynced_records(self, limit=50):
        # Keeping this for compatibility, though "synced" logic might change with direct DB connection
        # If writing directly to the central DB, everything is effectively "synced" instantly.
        # But we keep logic in case we want to flag rows or moved to a local buffer later.
        conn = self._get_connection()
        if not conn:
            return []

        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, timestamp, device_id, name
                FROM attendance_log
                WHERE synced = 0
                LIMIT %s
            ''', (limit,))
            return cursor.fetchall()
        except mysql.connector.Error as err:
             logger.error(f"Failed to get records: {err}")
             return []
        finally:
            if conn: conn.close()

    def mark_as_synced(self, record_ids):
        if not record_ids:
            return
        
        conn = self._get_connection()
        if not conn:
            return

        try:
            cursor = conn.cursor()
            placeholders = ', '.join(['%s'] * len(record_ids))
            cursor.execute(f'''
                UPDATE attendance_log
                SET synced = 1
                WHERE id IN ({placeholders})
            ''', record_ids)
            conn.commit()
        except mysql.connector.Error as err:
            logger.error(f"Failed to mark synced: {err}")
        finally:
            if conn: conn.close()
