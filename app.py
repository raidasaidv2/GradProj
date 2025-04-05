from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_from_directory, flash
import sqlite3
import bcrypt
import os
from werkzeug.utils import secure_filename
import cv2  # For video processing
import numpy as np  # For handling extracted features

app = Flask(__name__, static_folder='static')
app.secret_key = 'secret123'  # For session management

def verify_password(stored_password, provided_password):
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password)

def extract_gait_features(video_path):
    try:
        print(f"Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        features = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio = h / w if w > 0 else 0
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                features.append((aspect_ratio, cx, cy))

        cap.release()

        # Normalize the feature vector length to a fixed size (e.g., 100 frames)
        fixed_length = 100
        if len(features) > fixed_length:
            features = features[:fixed_length]  # Truncate to fixed length
        elif len(features) < fixed_length:
            padding = [(0, 0, 0)] * (fixed_length - len(features))  # Pad with zeros
            features.extend(padding)

        print(f"Extracted features (normalized): {features}")
        return features
    except Exception as e:
        print(f"Error during gait feature extraction: {e}")
        return None

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

    if request.method == 'POST':
        video = request.files['gait-video']

        if video:
            filename = secure_filename(video.filename)
            video_path = os.path.join('uploads', filename)
            video.save(video_path)
            print(f"Video saved to: {video_path}")

            # Extract gait features from the uploaded video
            features = extract_gait_features(video_path)
            print(f"Extracted features: {features}")

            if features is not None:
                conn = sqlite3.connect('gait_users.db')
                cursor = conn.cursor()

                # Fetch all employees and their gait features
                cursor.execute("SELECT name, gait_features FROM employees")
                employees = cursor.fetchall()
                conn.close()

                # Compare the extracted features with stored features
                closest_match = None
                closest_distance = float('inf')
                threshold = 50.0  # Set a threshold for identification

                for employee in employees:
                    name, stored_features = employee
                    stored_features = eval(stored_features)  # Convert string back to list

                    # Ensure stored features are normalized to the same length
                    if len(stored_features) > len(features):
                        stored_features = stored_features[:len(features)]
                    elif len(stored_features) < len(features):
                        padding = [(0, 0, 0)] * (len(features) - len(stored_features))
                        stored_features.extend(padding)

                    # Calculate Euclidean distance
                    distance = np.linalg.norm(np.array(features) - np.array(stored_features))
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_match = name

                # Check if the closest match is within the threshold
                if closest_distance <= threshold:
                    if "john" in closest_match.lower():
                        flash(f"Identified as: {closest_match}")
                    else:
                        flash(f"Person identified, but not John. Identified as: {closest_match}")
                else:
                    flash("Person not identified. Please try again.")
            else:
                flash("Failed to extract gait features. Please try again.")
        else:
            flash("Failed to upload video. Please try again.")

    return render_template('maindashboard.html')

@app.route('/activity-reports')
def activity_reports():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('activity-reports.html')

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

        # Extract gait features
        features = extract_gait_features(video_path)
        print(f"Extracted features: {features}")

        if features is not None:
            conn = sqlite3.connect('gait_users.db')
            cursor = conn.cursor()

            # Check if the emp_id already exists
            cursor.execute("SELECT * FROM employees WHERE emp_id = ?", (emp_id,))
            if cursor.fetchone():
                flash(f"Employee ID '{emp_id}' already exists. Please use a unique ID.")
                conn.close()
                return redirect(url_for('authorized_personnel'))

            # Save employee details and features to the database
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS employees (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    emp_id TEXT UNIQUE NOT NULL,
                    access_level TEXT NOT NULL,
                    gait_features TEXT NOT NULL
                )
            ''')
            cursor.execute('''
                INSERT INTO employees (name, emp_id, access_level, gait_features)
                VALUES (?, ?, ?, ?)
            ''', (name, emp_id, access_level, str(features)))
            conn.commit()
            conn.close()

            flash('Employee added successfully!')
        else:
            flash('Failed to extract gait features. Please try again.')
        return redirect(url_for('authorized_personnel'))
    else:
        flash('Failed to upload video. Please try again.')
        return redirect(url_for('authorized_personnel'))

@app.route('/identify', methods=['GET', 'POST'])
def identify():
    if 'username' not in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        video = request.files['gait-video']

        if video:
            filename = secure_filename(video.filename)
            video_path = os.path.join('uploads', filename)
            video.save(video_path)
            print(f"Video saved to: {video_path}")

            # Extract gait features from the uploaded video
            features = extract_gait_features(video_path)
            print(f"Extracted features: {features}")

            if features is not None:
                conn = sqlite3.connect('gait_users.db')
                cursor = conn.cursor()

                # Fetch all employees and their gait features
                cursor.execute("SELECT name, gait_features FROM employees")
                employees = cursor.fetchall()
                conn.close()

                # Compare the extracted features with stored features
                closest_match = None
                closest_distance = float('inf')

                for employee in employees:
                    name, stored_features = employee
                    stored_features = eval(stored_features)  # Convert string back to list

                    # Ensure stored features are normalized to the same length
                    if len(stored_features) > len(features):
                        stored_features = stored_features[:len(features)]
                    elif len(stored_features) < len(features):
                        padding = [(0, 0, 0)] * (len(features) - len(stored_features))
                        stored_features.extend(padding)

                    # Calculate Euclidean distance
                    distance = np.linalg.norm(np.array(features) - np.array(stored_features))
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_match = name

                if closest_match:
                    flash(f"Identified as: {closest_match}")
                else:
                    flash("No match found. Please try again.")
            else:
                flash("Failed to extract gait features. Please try again.")
        else:
            flash("Failed to upload video. Please try again.")

        return redirect(url_for('identify'))

    return render_template('identify.html')

@app.route('/static/<path:filename>')
def custom_static(filename):
    print(f"Serving static file: {filename}")
    return send_from_directory(app.static_folder, filename)

# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

if __name__ == '__main__':
    app.run(debug=True)