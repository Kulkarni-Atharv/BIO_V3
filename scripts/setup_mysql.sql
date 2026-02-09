CREATE DATABASE IF NOT EXISTS bio_attendance;

USE bio_attendance;

CREATE TABLE IF NOT EXISTS attendance_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DOUBLE,
    device_id VARCHAR(255),
    name VARCHAR(255),
    synced BOOLEAN DEFAULT 0
);
