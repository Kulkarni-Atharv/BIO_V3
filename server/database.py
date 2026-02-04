
import logging

logger = logging.getLogger("ServerDB")

try:
    import pyodbc
    PYODBC_AVAILABLE = True
except ImportError:
    PYODBC_AVAILABLE = False
    logger.warning("pyodbc not available (unixODBC missing?). Using Mock Database.")

class ServerDatabase:
    def __init__(self, connection_string=None):
        self.connection_string = connection_string
        if self.connection_string and PYODBC_AVAILABLE:
            self._init_db()
        else:
            logger.info("Initializing in Mock Mode (No SQL Server connection).")

    def _init_db(self):
        try:
            with pyodbc.connect(self.connection_string) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='AttendanceLogs' and xtype='U')
                    CREATE TABLE AttendanceLogs (
                        ID INT IDENTITY(1,1) PRIMARY KEY,
                        DeviceID NVARCHAR(50),
                        Name NVARCHAR(100),
                        Timestamp DATETIME,
                        ReceivedAt DATETIME DEFAULT GETDATE()
                    )
                ''')
                conn.commit()
                logger.info("Database initialized/verified.")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    def insert_attendance(self, device_id, name, timestamp):
        if not self.connection_string or not PYODBC_AVAILABLE:
            # Mock behavior for testing when no DB is configured
            logger.info(f"[MOCK DB] Insert: Device={device_id}, Name={name}, Time={timestamp}")
            return True

        try:
            with pyodbc.connect(self.connection_string) as conn:
                cursor = conn.cursor()
                from datetime import datetime
                dt = datetime.fromtimestamp(timestamp)
                
                cursor.execute("INSERT INTO AttendanceLogs (DeviceID, Name, Timestamp) VALUES (?, ?, ?)", 
                               (device_id, name, dt))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Database insert error: {e}")
            return False
