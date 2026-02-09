
import mysql.connector
import sys
import os

# Get absolute path to BIO/ and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.config import MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB, MYSQL_PORT

def test_connection():
    print(f"Testing connection to {MYSQL_HOST}...")
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB,
            port=MYSQL_PORT
        )
        if conn.is_connected():
            print("Successfully connected to MySQL database")
            
            cursor = conn.cursor()
            cursor.execute("SELECT DATABASE();")
            record = cursor.fetchone()
            print("You're connected to database: ", record)
            
            cursor.close()
            conn.close()
            print("MySQL connection is closed")
            return True
    except mysql.connector.Error as err:
        print(f"Error while connecting to MySQL: {err}")
        return False

if __name__ == "__main__":
    if test_connection():
        sys.exit(0)
    else:
        sys.exit(1)
