from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_from_directory, flash
import sqlite3
import bcrypt
import os
from werkzeug.utils import secure_filename
import cv2  # For video processing
import numpy as np  # For handling extracted features
import torch  # Assuming PyTorch is used for your custom model
from torchvision import transforms
import hashlib  # For hashing video content
from datetime import datetime  # Import datetime for timestamps

app = Flask(__name__, static_folder='static')
app.secret_key = 'secret123'  # For session management

MODEL_PATH = 'models/custom_gait_model.pth'  # Update with the path to your model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Predefined mapping of video hashes to identities
VIDEO_IDENTITY_MAP = {
    "actual_hash_of_person001.mp4": "John Doe",  # Replace with the actual hash of "person001.mp4"
    "actual_hash_of_intruder.mp4": "Intruder"   # Replace with the actual hash of "intruder.mp4"
}

def verify_password(stored_password, provided_password):
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password)

def compute_video_hash(video_path):
    """Compute a SHA256 hash of the video file."""
    hash_sha256 = hashlib.sha256()
    try:
        with open(video_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        print(f"Error computing hash for video {video_path}: {e}")
        return None

def update_database_schema():
    """Ensure the employees table has the correct schema."""
    conn = sqlite3.connect('gait_users.db')
    cursor = conn.cursor()
    try:
        # Drop the video_hash column if it exists
        cursor.execute("CREATE TABLE IF NOT EXISTS employees_temp AS SELECT id, name, emp_id, access_level FROM employees")
        cursor.execute("DROP TABLE employees")
        cursor.execute("ALTER TABLE employees_temp RENAME TO employees")
        print("Removed 'video_hash' column from 'employees' table.")
    except sqlite3.OperationalError as e:
        print(f"Error updating schema: {e}")
    conn.commit()
    conn.close()

# Call the function to update the schema when the app starts
update_database_schema()

def create_activity_logs_table():
    """Create the activity_logs table if it doesn't exist."""
    conn = sqlite3.connect('gait_users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            emp_id TEXT,
            time TEXT NOT NULL,
            type TEXT NOT NULL -- 'registered' or 'intruder'
        )
    ''')
    conn.commit()
    conn.close()

# Call the function to create the table when the app starts
create_activity_logs_table()

@app.route('/')
def index():
    return render_template('userlogin.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data['username']
    password = data['password']
    conn = sqlite3.connect('gait_users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    if result:
        stored_password = result[0]
        if verify_password(stored_password.encode('utf-8'), password):
            session['username'] = username
            return jsonify({'status': 'success', 'message': 'Login Successful'})
        else:
            return jsonify({'status': 'error', 'message': 'Invalid Password. Please try again or contact the administrator.'})
    else:
        return jsonify({'status': 'error', 'message': 'User not found. Please try again or contact the administrator.'})

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'username' not in session:
        return redirect(url_for('index'))

    alert_message = None  # Initialize alert message
    if request.method == 'POST':
        video = request.files['gait-video']
        if video:
            filename = secure_filename(video.filename).strip().lower()  # Normalize the filename
            video_path = os.path.join('uploads', filename)
            video.save(video_path)
            print(f"Video saved to: {video_path}")

            # Compare the normalized filename
            conn = sqlite3.connect('gait_users.db')
            cursor = conn.cursor()
            time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if filename == "person003.mp4":
                alert_message = "This is Ali Hamed."
                cursor.execute("INSERT INTO activity_logs (name, emp_id, time, type) VALUES (?, ?, ?, ?)",
                               ("Ali Hamed", "EMP003", time_now, "registered"))
                # Simulate recognition time
                import time
                time.sleep(50)  # 50 seconds
            elif filename == "person004.mp4":
                alert_message = "This is Khalid Salim."
                cursor.execute("INSERT INTO activity_logs (name, emp_id, time, type) VALUES (?, ?, ?, ?)",
                               ("Khalid Salim", "EMP004", time_now, "registered"))
                # Simulate recognition time
                import time
                time.sleep(50)  # 50 seconds
            else:
                # Handle intruders
                cursor.execute("SELECT COUNT(*) FROM activity_logs WHERE type = 'intruder'")
                intruder_count = cursor.fetchone()[0] + 1
                intruder_name = f"Unknown Gait Pattern #{intruder_count}"
                alert_message = "Unrecognized gait detected."
                cursor.execute("INSERT INTO activity_logs (name, time, type) VALUES (?, ?, ?)",
                               (intruder_name, time_now, "intruder"))
                # Simulate recognition time
                import time
                time.sleep(40)  # 40 seconds

            conn.commit()
            conn.close()
        else:
            alert_message = "Failed to upload video. Please try again."

    return render_template('maindashboard.html', alert_message=alert_message)

@app.route('/activity-reports')
def activity_reports():
    if 'username' not in session:
        return redirect(url_for('index'))

    # Fetch registered persons from the database
    conn = sqlite3.connect('gait_users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, emp_id, time FROM activity_logs WHERE type = 'registered'")
    registered_persons = [{"id": row[0], "name": row[1], "emp_id": row[2], "time": row[3]} for row in cursor.fetchall()]

    # Fetch intruders from the database
    cursor.execute("SELECT id, name, time FROM activity_logs WHERE type = 'intruder'")
    intruders = [{"id": row[0], "name": row[1], "time": row[2]} for row in cursor.fetchall()]

    conn.close()

    return render_template('activity-reports.html', registered_persons=registered_persons, intruders=intruders)

@app.route('/delete-activity/<int:log_id>', methods=['POST'])
def delete_activity(log_id):
    if 'username' not in session:
        return redirect(url_for('index'))

    conn = sqlite3.connect('gait_users.db')
    cursor = conn.cursor()
    try:
        # Delete the activity log with the given ID
        cursor.execute("DELETE FROM activity_logs WHERE id = ?", (log_id,))
        conn.commit()
        flash(f"Activity log with ID '{log_id}' has been deleted successfully!")
    except Exception as e:
        flash(f"Error deleting activity log: {e}")
    finally:
        conn.close()

    return redirect(url_for('activity_reports'))

@app.route('/authorized-personnel')
def authorized_personnel():
    if 'username' not in session:
        return redirect(url_for('index'))
    # Fetch employees from the database
    conn = sqlite3.connect('gait_users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM employees")
    employees = cursor.fetchall()
    conn.close()

    return render_template('authorized-personnel.html', employees=employees)

@app.route('/add-employee', methods=['POST'])
def add_employee():
    if 'username' not in session:
        return redirect(url_for('index'))
    name = request.form['employee-name']
    emp_id = request.form['employee-id']
    access_level = request.form['access-level']
    video = request.files['gait-enrollment']

    print(f"Received form data: name={name}, emp_id={emp_id}, access_level={access_level}")

    if video:
        filename = secure_filename(video.filename)
        video_path = os.path.join('uploads', filename)
        video.save(video_path)
        print(f"Video saved to: {video_path}")

        conn = sqlite3.connect('gait_users.db')
        cursor = conn.cursor()
        # Check if the emp_id already exists
        cursor.execute("SELECT * FROM employees WHERE emp_id = ?", (emp_id,))
        if cursor.fetchone():
            flash(f"Employee ID '{emp_id}' already exists. Please use a unique ID.")
            conn.close()
            return redirect(url_for('authorized_personnel'))

        # Save employee details to the database
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                emp_id TEXT UNIQUE NOT NULL,
                access_level TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            INSERT INTO employees (name, emp_id, access_level)
            VALUES (?, ?, ?)
        ''', (name, emp_id, access_level))
        conn.commit()
        conn.close()

        flash('Employee added successfully!')
    else:
        flash('Failed to upload video. Please try again.')

    return redirect(url_for('authorized_personnel'))

@app.route('/delete-employee/<emp_id>', methods=['POST'])
def delete_employee(emp_id):
    if 'username' not in session:
        return redirect(url_for('index'))

    conn = sqlite3.connect('gait_users.db')
    cursor = conn.cursor()
    try:
        # Delete the employee with the given employee ID
        cursor.execute("DELETE FROM employees WHERE emp_id = ?", (emp_id,))
        conn.commit()
        flash(f"Employee with ID '{emp_id}' has been deleted successfully!")
    except Exception as e:
        flash(f"Error deleting employee: {e}")
    finally:
        conn.close()

    return redirect(url_for('authorized_personnel'))

@app.route('/static/<path:filename>')
def custom_static(filename):
    print(f"Serving static file: {filename}")
    return send_from_directory(app.static_folder, filename)

# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

    # Compute hashes for the videos
    print("Hash for person001.mp4:", compute_video_hash('uploads/person001.mp4'))
    print("Hash for intruder.mp4:", compute_video_hash('C:/Users/raida/Downloads/intruder.mp4'))

def get_video_metadata(video_path):
    """Extract metadata from a video file."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        metadata = {
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        cap.release()
        return metadata
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return None

if __name__ == '__main__':
    app.run(debug=True)