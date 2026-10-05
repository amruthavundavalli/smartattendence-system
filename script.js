document.addEventListener("DOMContentLoaded", function () {
    // Automatically hide success and error messages after 5 seconds.
    const messages = document.querySelectorAll(".message");

    messages.forEach(function (message) {
        setTimeout(function () {
            message.style.display = "none";
        }, 5000);
    });

    // Ask for confirmation before submitting the attendance form.
    const attendanceForm = document.querySelector('form[action="/attendance"]');

    if (attendanceForm) {
        attendanceForm.addEventListener("submit", function (event) {
            const studentSelect = document.getElementById("student_id");

            if (studentSelect && !studentSelect.value) {
                event.preventDefault();
                alert("Please select a student first.");
            }
        });
    }
});