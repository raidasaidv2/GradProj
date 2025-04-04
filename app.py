from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_from_directory, flash
import sqlite3
import bcrypt
import os
from werkzeug.utils import secure_filename
import cv2  # For video processing
import numpy as np  # For handling extracted features
from opengait.models import load_model
from opengait.utils import preprocess_video

app = Flask(__name__, static_folder='static')
app.secret_key = 'secret123'  # For session management

def verify_password(stored_password, provided_password):
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password)

# Updated extract_gait_features function
def extract_gait_features(video_path):
    try:
        # Preprocess the video into aligned silhouettes
        sils = preprocess_video(video_path)

        # Load the pre-trained OpenGait model
        model = load_model('path_to_pretrained_model.pth')

        # Extract gait features
        features = model.infer(sils)
        return features.tolist()
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

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('index'))
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
    return render_template('authorized-personnel.html')

@app.route('/add-employee', methods=['POST'])
def add_employee():
    if 'username' not in session:
        return redirect(url_for('index'))

    name = request.form['employee-name']
    emp_id = request.form['employee-id']
    access_level = request.form['access-level']
    video = request.files['gait-enrollment']

    if video:
        filename = secure_filename(video.filename)
        video_path = os.path.join('uploads', filename)
        video.save(video_path)

        # Extract gait features
        features = extract_gait_features(video_path)

        if features is not None:
            # Save employee details and features to the database
            conn = sqlite3.connect('gait_users.db')
            cursor = conn.cursor()
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

@app.route('/static/<path:filename>')
def custom_static(filename):
    print(f"Serving static file: {filename}")
    return send_from_directory(app.static_folder, filename)

# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

if __name__ == '__main__':
    app.run(debug=True)