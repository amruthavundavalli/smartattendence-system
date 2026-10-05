import sqlite3
from datetime import datetime, date

DATABASE_NAME = "attendance.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_NAME)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            roll_number TEXT NOT NULL UNIQUE,
            department TEXT,
            email TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Present',
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    """)

    connection.commit()
    connection.close()


def add_student(name, roll_number, department="", email=""):
    name = str(name).strip()
    roll_number = str(roll_number).strip().upper()
    department = str(department).strip()
    email = str(email).strip()

    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute("""
            INSERT INTO students (
                name,
                roll_number,
                department,
                email
            )
            VALUES (?, ?, ?, ?)
        """, (
            name,
            roll_number,
            department,
            email
        ))

        connection.commit()

        return {
            "success": True,
            "message": "Student added successfully."
        }

    except sqlite3.IntegrityError:
        connection.rollback()

        return {
            "success": False,
            "message": f"The roll number '{roll_number}' already exists."
        }

    except Exception as error:
        connection.rollback()

        return {
            "success": False,
            "message": f"Database error: {error}"
        }

    finally:
        connection.close()


def get_all_students():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            name,
            roll_number,
            department,
            email
        FROM students
        ORDER BY id DESC
    """)

    students = cursor.fetchall()

    connection.close()

    return students


def get_student_by_id(student_id):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            name,
            roll_number,
            department,
            email
        FROM students
        WHERE id = ?
    """, (student_id,))

    student = cursor.fetchone()

    connection.close()

    return student


def mark_attendance(student_id):
    connection = get_connection()
    cursor = connection.cursor()

    today = date.today().isoformat()
    current_time = datetime.now().strftime("%H:%M:%S")

    cursor.execute("""
        SELECT id
        FROM attendance
        WHERE student_id = ?
        AND date = ?
        LIMIT 1
    """, (student_id, today))

    existing_record = cursor.fetchone()

    if existing_record:
        connection.close()
        return False

    cursor.execute("""
        INSERT INTO attendance (
            student_id,
            date,
            time,
            status
        )
        VALUES (?, ?, ?, ?)
    """, (
        student_id,
        today,
        current_time,
        "Present"
    ))

    connection.commit()
    connection.close()

    return True


def get_attendance_records():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            attendance.id,
            students.name,
            students.roll_number,
            students.department,
            attendance.date,
            attendance.time,
            attendance.status
        FROM attendance
        INNER JOIN students
            ON attendance.student_id = students.id
        ORDER BY
            attendance.date DESC,
            attendance.time DESC
    """)

    records = cursor.fetchall()

    connection.close()

    return records


def get_dashboard_stats():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT COUNT(*) AS total_students
        FROM students
    """)

    total_students = cursor.fetchone()["total_students"]

    today = date.today().isoformat()

    cursor.execute("""
        SELECT COUNT(DISTINCT student_id) AS present_today
        FROM attendance
        WHERE date = ?
        AND status = 'Present'
    """, (today,))

    present_today = cursor.fetchone()["present_today"]

    if total_students > 0:
        attendance_rate = round(
            (present_today / total_students) * 100,
            1
        )
    else:
        attendance_rate = 0

    connection.close()

    return {
        "total_students": total_students,
        "present_today": present_today,
        "attendance_rate": attendance_rate
    }


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")