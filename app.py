from flask import Flask, render_template, request, redirect, url_for, flash

from database import (
    init_db,
    get_dashboard_stats,
    get_all_students,
    add_student,
    get_attendance_records,
    mark_attendance
)

from face_recognition_module import (
    capture_student_photos,
    train_recognizer,
    start_photo_attendance
)

app = Flask(__name__)
app.secret_key = "smart-attendance-project-key"

init_db()


@app.route("/")
def index():
    stats = get_dashboard_stats()
    return render_template("index.html", stats=stats)


@app.route("/students")
def students():
    all_students = get_all_students()
    return render_template("students.html", students=all_students)


@app.route("/add-student", methods=["GET", "POST"])
def add_student_page():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        roll_number = request.form.get("roll_number", "").strip()
        department = request.form.get("department", "").strip()
        email = request.form.get("email", "").strip()

        if not name or not roll_number:
            flash("Name and roll number are required.", "error")
            return redirect(url_for("add_student_page"))

        try:
            add_student(name, roll_number, department, email)
            flash("Student added successfully.", "success")
            return redirect(url_for("students"))
        except Exception:
            flash("That roll number may already exist.", "error")

    return render_template("add_student.html")


@app.route("/enroll/<int:student_id>", methods=["POST"])
def enroll_student(student_id):
    success = capture_student_photos(student_id)

    if success:
        flash(
            "Student photos captured. Train the face model before attendance.",
            "success"
        )
    else:
        flash("Photo enrollment was not completed.", "error")

    return redirect(url_for("students"))


@app.route("/train-faces", methods=["POST"])
def train_faces():
    if train_recognizer():
        flash("Face recognition model trained successfully.", "success")
    else:
        flash("Could not train the model. Enroll student photos first.", "error")

    return redirect(url_for("students"))


@app.route("/photo-attendance", methods=["GET", "POST"])
def photo_attendance():
    if request.method == "POST":
        start_photo_attendance()
        return redirect(url_for("attendance"))

    return render_template("photo_attendance.html")


@app.route("/attendance", methods=["GET", "POST"])
def attendance():
    if request.method == "POST":
        student_id = request.form.get("student_id", "").strip()

        if student_id.isdigit():
            mark_attendance(int(student_id))
            flash("Attendance recorded.", "success")
        else:
            flash("Please select a valid student.", "error")

        return redirect(url_for("attendance"))

    all_students = get_all_students()
    records = get_attendance_records()

    return render_template(
        "attendance.html",
        students=all_students,
        records=records
    )


@app.route("/reports")
def reports():
    records = get_attendance_records()
    return render_template("reports.html", records=records)


if __name__ == "__main__":
    app.run(debug=True)