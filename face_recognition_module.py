import json
from pathlib import Path

import cv2
import numpy as np

from database import get_all_students, mark_attendance


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

DATASET_DIR = Path("face_dataset")
MODEL_DIR = Path("face_model")

MODEL_PATH = MODEL_DIR / "face_model.yml"
LABELS_PATH = MODEL_DIR / "labels.json"

FACE_SIZE = (200, 200)
SAMPLES_TO_CAPTURE = 30

# Lower number = stricter recognition
CONFIDENCE_THRESHOLD = 75


# ---------------------------------------------------------
# FACE DETECTOR
# ---------------------------------------------------------

def get_face_detector():
    """Create the OpenCV Haar face detector."""

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError(
            "Could not load the OpenCV face detector."
        )

    return detector


# ---------------------------------------------------------
# CAMERA
# ---------------------------------------------------------

def open_camera():
    """
    Open the Windows camera reliably.

    Tries DirectShow first, then other camera indexes.
    """

    camera_indexes = [0, 1, 2]

    for index in camera_indexes:

        try:
            print(f"Trying camera {index}...")

            camera = cv2.VideoCapture(
                index,
                cv2.CAP_DSHOW
            )

            if camera.isOpened():

                # Give the camera a moment to initialize.
                for _ in range(10):
                    success, frame = camera.read()

                    if success and frame is not None:
                        print(
                            f"Camera {index} opened successfully."
                        )
                        return camera

                camera.release()

        except Exception as error:
            print(
                f"Camera {index} error: {error}"
            )

    print()
    print("Could not open any camera.")
    print("Please check that:")
    print("1. Your camera is connected.")
    print("2. No other application is using the camera.")
    print("3. Windows Camera permission is enabled.")
    print("4. Zoom, Teams, WhatsApp or another camera app is closed.")

    return None


# ---------------------------------------------------------
# CAPTURE STUDENT PHOTOS
# ---------------------------------------------------------

def capture_student_photos(student_id):
    """
    Capture face samples for one registered student.

    The student must give permission before enrollment.
    """

    students = get_all_students()

    student = next(
        (
            item
            for item in students
            if item["id"] == student_id
        ),
        None
    )

    if student is None:
        print("Student not found.")
        return False

    student_dir = DATASET_DIR / str(student_id)
    student_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    detector = get_face_detector()

    camera = open_camera()

    if camera is None:
        return False

    saved_count = 0

    print()
    print("----------------------------------------")
    print("PHOTO ENROLLMENT")
    print("----------------------------------------")
    print(f"Student: {student['name']}")
    print("Look directly at the camera.")
    print("Move your head slightly left and right.")
    print("Press Q or ESC to stop.")
    print("----------------------------------------")
    print()

    try:

        while saved_count < SAMPLES_TO_CAPTURE:

            success, frame = camera.read()

            if not success or frame is None:
                print("Could not read from the camera.")
                break

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80)
            )

            # Draw all detected faces.
            for x, y, w, h in faces:

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

            # Instruction.
            cv2.putText(
                frame,
                "SMART ATTENDANCE",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Photos: {saved_count}/{SAMPLES_TO_CAPTURE}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Look at camera | Q/ESC = Stop",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            if len(faces) == 1:

                x, y, w, h = faces[0]

                face = gray[
                    y:y + h,
                    x:x + w
                ]

                if face.size > 0:

                    face = cv2.resize(
                        face,
                        FACE_SIZE
                    )

                    file_path = (
                        student_dir
                        / f"{saved_count + 1}.jpg"
                    )

                    saved = cv2.imwrite(
                        str(file_path),
                        face
                    )

                    if saved:

                        saved_count += 1

                        # Small delay between photos.
                        cv2.waitKey(150)

            elif len(faces) > 1:

                cv2.putText(
                    frame,
                    "Only ONE face should be visible",
                    (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "No face detected",
                    (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(
                "Student Photo Enrollment",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    except Exception as error:

        print(
            f"Photo enrollment error: {error}"
        )

    finally:

        camera.release()
        cv2.destroyAllWindows()

    print()
    print(
        f"Saved {saved_count} photos for "
        f"{student['name']}."
    )

    return saved_count >= 5


# ---------------------------------------------------------
# TRAIN FACE MODEL
# ---------------------------------------------------------

def train_recognizer():
    """
    Train the LBPH face recognition model
    using enrolled student photos.
    """

    if not hasattr(cv2, "face"):

        print("OpenCV face module is missing.")
        print(
            "Make sure opencv-contrib-python "
            "is installed."
        )

        return False

    if not DATASET_DIR.exists():

        print(
            "No face dataset found."
        )

        return False

    faces = []
    labels = []

    label_to_student = {}

    # Go through each student's folder.
    for student_dir in DATASET_DIR.iterdir():

        if not student_dir.is_dir():
            continue

        try:
            student_id = int(
                student_dir.name
            )

        except ValueError:
            continue

        image_files = list(
            student_dir.glob("*.jpg")
        )

        for image_path in image_files:

            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                continue

            image = cv2.resize(
                image,
                FACE_SIZE
            )

            faces.append(image)
            labels.append(student_id)

            label_to_student[
                str(student_id)
            ] = student_id

    if not faces:

        print(
            "No enrollment photos found."
        )

        return False

    print()
    print(
        f"Training with {len(faces)} photos..."
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    recognizer = (
        cv2.face.LBPHFaceRecognizer_create()
    )

    recognizer.train(
        faces,
        np.array(labels)
    )

    recognizer.write(
        str(MODEL_PATH)
    )

    with open(
        LABELS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            label_to_student,
            file,
            indent=4
        )

    print(
        "Face model trained successfully."
    )

    return True


# ---------------------------------------------------------
# PHOTO ATTENDANCE
# ---------------------------------------------------------

def start_photo_attendance():
    """
    Recognize enrolled students through the camera.

    Press Q or ESC to close the camera.
    """

    if not hasattr(cv2, "face"):

        print(
            "OpenCV face module is missing."
        )

        return

    if not MODEL_PATH.exists():

        print(
            "No trained face model found."
        )

        print(
            "Enroll students and train "
            "the model first."
        )

        return

    if not LABELS_PATH.exists():

        print(
            "Face labels file is missing."
        )

        return

    students = get_all_students()

    student_by_id = {
        student["id"]: student
        for student in students
    }

    recognizer = (
        cv2.face.LBPHFaceRecognizer_create()
    )

    try:

        recognizer.read(
            str(MODEL_PATH)
        )

    except Exception as error:

        print(
            f"Could not load face model: {error}"
        )

        return

    detector = get_face_detector()

    camera = open_camera()

    if camera is None:
        return

    print()
    print("----------------------------------------")
    print("PHOTO ATTENDANCE")
    print("----------------------------------------")
    print("Look at the camera.")
    print("Press Q or ESC to stop.")
    print("----------------------------------------")
    print()

    # Prevent marking the same person repeatedly.
    marked_students = set()

    try:

        while True:

            success, frame = camera.read()

            if not success or frame is None:

                print(
                    "Could not read from the camera."
                )

                break

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80)
            )

            for x, y, w, h in faces:

                face = gray[
                    y:y + h,
                    x:x + w
                ]

                if face.size == 0:
                    continue

                face = cv2.resize(
                    face,
                    FACE_SIZE
                )

                try:

                    predicted_id, confidence = (
                        recognizer.predict(face)
                    )

                except Exception:

                    continue

                student = student_by_id.get(
                    predicted_id
                )

                # LBPH confidence is a distance:
                # LOWER is generally better.
                if (
                    student is not None
                    and confidence
                    <= CONFIDENCE_THRESHOLD
                ):

                    name = student["name"]

                    label = (
                        f"{name} - PRESENT"
                    )

                    box_color = (
                        0,
                        255,
                        0
                    )

                    # Mark only once per camera session.
                    if predicted_id not in marked_students:

                        mark_attendance(
                            predicted_id
                        )

                        marked_students.add(
                            predicted_id
                        )

                        print(
                            f"Attendance marked: "
                            f"{name}"
                        )

                else:

                    label = "UNKNOWN"

                    box_color = (
                        0,
                        0,
                        255
                    )

                # Face rectangle.
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    box_color,
                    3
                )

                # Name / unknown label.
                cv2.putText(
                    frame,
                    label,
                    (
                        x,
                        max(y - 10, 25)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    box_color,
                    2
                )

                # Confidence display.
                cv2.putText(
                    frame,
                    f"Score: {confidence:.1f}",
                    (
                        x,
                        y + h + 25
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box_color,
                    1
                )

            # Header.
            cv2.putText(
                frame,
                "SMART ATTENDANCE",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Q / ESC = Stop",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "Photo Attendance",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if (
                key == ord("q")
                or key == 27
            ):
                break

    except Exception as error:

        print(
            f"Photo attendance error: {error}"
        )

    finally:

        camera.release()
        cv2.destroyAllWindows()

    print()
    print(
        "Photo attendance stopped."
    )


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    print(
        "Face recognition module is ready."
    )

    print(
        "Use the website to enroll students "
        "and start photo attendance."
    )