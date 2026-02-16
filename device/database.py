import mysql.connector
import time
import os
import sys
import logging
from datetime import datetime, timedelta, date, time as dt_time

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
        """Initializes the database schema with support for Shifts and detailed Attendance Logs."""
        conn = self._get_connection()
        if not conn:
            logger.error("Failed to connect to MySQL Database during initialization.")
            return

        try:
            cursor = conn.cursor()
            
            # 1. Create Shifts Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS shifts (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    shift_name VARCHAR(50) NOT NULL,
                    start_time TIME NOT NULL,
                    end_time TIME NOT NULL,
                    late_grace_mins INT DEFAULT 15,
                    half_day_min_hours FLOAT DEFAULT 4.0,
                    overtime_start_mins INT DEFAULT 30,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Check if default shift exists, if not create one (General Shift: 09:00 - 18:00)
            cursor.execute("SELECT COUNT(*) FROM shifts")
            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                    INSERT INTO shifts (shift_name, start_time, end_time, late_grace_mins)
                    VALUES ('General Shift', '09:00:00', '18:00:00', 15)
                """)
                logger.info("Created default 'General Shift'.")

            # 2. Create/Update Attendance Log Table
            # We use the user's requested schema structure + some additions for processing
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance_log (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id VARCHAR(100),
                    name VARCHAR(255),
                    device_id VARCHAR(100),
                    punch_time DATETIME,
                    punch_date DATE,
                    punch_clock TIME,
                    punch_type ENUM('IN','OUT'),
                    shift_id INT,
                    attendance_status VARCHAR(50) DEFAULT 'Present',
                    late_minutes INT DEFAULT 0,
                    early_departure_minutes INT DEFAULT 0,
                    overtime_minutes INT DEFAULT 0,
                    confidence FLOAT,
                    synced BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (shift_id) REFERENCES shifts(id)
                )
            """)

            conn.commit()
            conn.commit()
            
            # --- SCHEMA MIGRATION ---
            # Check if 'attendance_log' has 'user_id' column
            cursor.execute("DESCRIBE attendance_log")
            columns = [row[0] for row in cursor.fetchall()]
            
            if 'user_id' not in columns:
                logger.info("Migrating schema: Adding missing columns to 'attendance_log'...")
                try:
                    # Add new columns individually or in bulk
                    # We use IGNORE or check, but since we know user_id is missing, we add the batch
                    cursor.execute("""
                        ALTER TABLE attendance_log
                        ADD COLUMN user_id VARCHAR(100) AFTER id,
                        ADD COLUMN punch_date DATE AFTER punch_time,
                        ADD COLUMN punch_clock TIME AFTER punch_date,
                        ADD COLUMN punch_type ENUM('IN','OUT') AFTER punch_clock,
                        ADD COLUMN shift_id INT AFTER punch_type,
                        ADD COLUMN attendance_status VARCHAR(50) DEFAULT 'Present' AFTER shift_id,
                        ADD COLUMN late_minutes INT DEFAULT 0 AFTER attendance_status,
                        ADD COLUMN early_departure_minutes INT DEFAULT 0 AFTER late_minutes,
                        ADD COLUMN overtime_minutes INT DEFAULT 0 AFTER early_departure_minutes,
                        ADD COLUMN confidence FLOAT AFTER overtime_minutes,
                        ADD CONSTRAINT fk_shift FOREIGN KEY (shift_id) REFERENCES shifts(id)
                    """)
                    conn.commit()
                    logger.info("Schema migration completed successfully.")
                except mysql.connector.Error as err:
                     logger.error(f"Schema Migration Error: {err}")

            logger.info("Database schema initialized successfully.")

        except mysql.connector.Error as err:
            logger.error(f"Database Initialization Error: {err}")
        finally:
            if conn: conn.close()

    def get_user_shift(self, user_id):
        """
        Retrieves the shift for a user. 
        For now, returns the Default General Shift (ID 1) or the first available shift.
        In a real app, you'd have a user_shifts table mapping users to shifts.
        """
        conn = self._get_connection()
        if not conn: return None
        
        try:
            cursor = conn.cursor(dictionary=True)
            # Placeholder: Just get the first shift
            cursor.execute("SELECT * FROM shifts ORDER BY id ASC LIMIT 1")
            shift = cursor.fetchone()
            return shift
        except Exception as e:
            logger.error(f"Error getting shift: {e}")
            return None
        finally:
            if conn: conn.close()
    
    def get_last_punch_today(self, user_id):
        """Get the last punch for the user today to determine IN/OUT sequence."""
        conn = self._get_connection()
        if not conn: return None
        
        try:
            cursor = conn.cursor(dictionary=True)
            today = date.today()
            cursor.execute("""
                SELECT * FROM attendance_log 
                WHERE user_id = %s AND punch_date = %s 
                ORDER BY punch_time DESC LIMIT 1
            """, (user_id, today))
            return cursor.fetchone()
        except Exception as e:
            logger.error(f"Error getting last punch: {e}")
            return None
        finally:
            if conn: conn.close()

    def calculate_attendance_status(self, punch_time, punch_type, shift):
        """
        Calculates Late, Early Leaving, Overtime based on Shift Rules.
        Returns: (status, late_mins, early_mins, ot_mins)
        """
        if not shift:
            return "Present", 0, 0, 0

        status = "Present"
        late_mins = 0
        early_mins = 0
        ot_mins = 0

        # Convert times to datetime for comparison (using today's date)
        # Note: This simple logic assumes day shifts (same day start/end). 
        # Night shifts crossing midnight need more complex date handling.
        base_date = punch_time.date()
        
        # Parse Shift Times
        # start_time is timedalta or time object depending on connector version
        # We ensure they are standard datetime objects on base_date
        
        def to_dt(t_obj):
            if isinstance(t_obj, timedelta):
                # Convert timedelta to time
                 total_seconds = int(t_obj.total_seconds())
                 hours = total_seconds // 3600
                 minutes = (total_seconds % 3600) // 60
                 return datetime.combine(base_date, dt_time(hours, minutes))
            elif isinstance(t_obj, dt_time):
                 return datetime.combine(base_date, t_obj)
            else:
                 # Check if string
                 return datetime.combine(base_date, datetime.strptime(str(t_obj), "%H:%M:%S").time())

        shift_start = to_dt(shift['start_time'])
        shift_end = to_dt(shift['end_time'])
        
        current_time = punch_time
        
        if punch_type == 'IN':
            # Check for Late Entry
            # Grace period
            late_threshold = shift_start + timedelta(minutes=shift['late_grace_mins'])
            
            if current_time > late_threshold:
                status = "Late"
                # Calculate late minutes
                diff = current_time - shift_start
                late_mins = int(diff.total_seconds() / 60)
                
                # Check for Half Day (if verify late)
                # If they are late by more than half the shift duration? 
                # Or simplistic: Late by > 2 hours = Half Day?
                # User req: "late mark half day" - maybe if late > X mins?
                # Using hardcoded logic for now: if late > 2 hours -> Half Day
                if late_mins > 120:
                     status = "Half Day"

        elif punch_type == 'OUT':
            # Check for Early Departure or Overtime
            if current_time < shift_end:
                 # Early
                 diff = shift_end - current_time
                 early_mins = int(diff.total_seconds() / 60)
                 # If early by > 1 hour -> Half Day?
                 if early_mins > 60:
                     status = "Half Day (Early)"
                 else:
                     status = "Early Departure"
            
            elif current_time > (shift_end + timedelta(minutes=shift['overtime_start_mins'])):
                # Overtime
                diff = current_time - shift_end
                ot_mins = int(diff.total_seconds() / 60)
                status = "Overtime"
        
        return status, late_mins, early_mins, ot_mins

    def add_record(self, device_id, name, user_id=None, confidence=0.0):
        timestamp = time.time()
        dt_now = datetime.fromtimestamp(timestamp)
        p_date = dt_now.date()
        p_time = dt_now.time()
        
        # Temporary user_id generation if not provided (Migration support)
        if not user_id:
            user_id = name # Use name as ID if missing
            
        conn = self._get_connection()
        if not conn: return None
        
        try:
            # 1. Determine Punch Type (Auto-toggle)
            last_punch = self.get_last_punch_today(user_id)
            if not last_punch:
                punch_type = 'IN'
            else:
                # If last was IN, now is OUT. If last was OUT, now is IN (Multiple entries allowed)
                last_type = last_punch['punch_type']
                punch_type = 'OUT' if last_type == 'IN' else 'IN'
            
            # 2. Get Shift Info
            shift = self.get_user_shift(user_id)
            shift_id = shift['id'] if shift else None
            
            # 3. Calculate Status
            status, late, early, ot = self.calculate_attendance_status(dt_now, punch_type, shift)
            
            cursor = conn.cursor()
            query = """
                INSERT INTO attendance_log 
                (user_id, name, device_id, punch_time, punch_date, punch_clock, punch_type, 
                 shift_id, attendance_status, late_minutes, early_departure_minutes, overtime_minutes, confidence, synced)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 0)
            """
            values = (
                user_id, name, device_id, dt_now, p_date, p_time, punch_type,
                shift_id, status, late, early, ot, confidence
            )
            
            cursor.execute(query, values)
            conn.commit()
            
            logger.info(f"Recorded {punch_type} for {name}: Status={status}, Late={late}m, OT={ot}m")
            return cursor.lastrowid

        except mysql.connector.Error as err:
            logger.error(f"Failed to add record: {err}")
            return None
        finally:
            if conn: conn.close()

    def get_unsynced_records(self, limit=50):
        conn = self._get_connection()
        if not conn: return []

        try:
            cursor = conn.cursor(dictionary=True) # Dictionary cursor for easier handling
            cursor.execute('''
                SELECT * FROM attendance_log
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
        if not record_ids: return
        conn = self._get_connection()
        if not conn: return

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
