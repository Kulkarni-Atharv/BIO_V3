-- Database Selection/Creation
CREATE DATABASE IF NOT EXISTS bio_attendance;
USE bio_attendance;

-- 1. Shifts Table
-- Stores shift definitions with grace periods and overtime rules
CREATE TABLE IF NOT EXISTS shifts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    shift_name VARCHAR(50) NOT NULL,
    start_time TIME NOT NULL,
    end_time TIME NOT NULL,
    late_grace_mins INT DEFAULT 15,
    half_day_min_hours FLOAT DEFAULT 4.0,
    overtime_start_mins INT DEFAULT 30,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert Default General Shift if not exists
INSERT IGNORE INTO shifts (id, shift_name, start_time, end_time, late_grace_mins) 
VALUES (1, 'General Shift', '09:00:00', '18:00:00', 15);

-- 2. Attendance Log Table
-- Stores real-time punch data with calculated statuses
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
);

-- Optional: Create an index for faster date-range queries (Monitoring)
CREATE INDEX idx_punch_date ON attendance_log(punch_date);
CREATE INDEX idx_user_id ON attendance_log(user_id);
