# GaitSecure: Gait-Based Identification System

## Overview
GaitSecure is a gait-based identification system designed to recognize individuals and detect intruders using video analysis. The system leverages machine learning models to analyze gait patterns and classify individuals as either registered personnel or intruders. It provides a user-friendly interface for managing authorized personnel, viewing activity reports, and identifying individuals in real-time.

## Features
- **User Authentication**: Secure login system for accessing the dashboard.
- **Real-Time Identification**: Upload gait videos to identify registered personnel or detect intruders.
- **Activity Reports**: View detailed logs of registered personnel and intruder activities.
- **Authorized Personnel Management**: Add, view, and delete registered personnel.
- **Visual Analytics**:
  - Confusion matrix and classification reports for model performance evaluation.
  - ROC curve and feature importance visualization for machine learning insights.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/GradProj.git
   cd GradProj
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the following files are placed in the appropriate directories:
   - `gait_model.pkl`: Trained machine learning model.
   - `label_encoder.pkl`: Label encoder for class names.
   - `gait_dataset.pkl`: Dataset for evaluation.

4. Run the Flask application:
   ```bash
   python app.py
   ```

5. Access the application in your browser at `http://127.0.0.1:5000`.

## Project Structure
```
GradProj/
├── static/
│   ├── images/                    # Images for the UI
│   ├── maindashboardstyling.css   # CSS for the main dashboard
│   ├── userguistyling.css         # CSS for the login page
│   ├── activityreports.css        # CSS for activity reports
├── templates/
│   ├── userlogin.html             # Login page
│   ├── maindashboard.html         # Main dashboard
│   ├── authorized-personnel.html  # Authorized personnel management
│   ├── activity-reports.html      # Activity reports
│   ├── identify.html              # Identify person page
│   ├── error.html                 # Error page for invalid actions
├── models/
│   ├── gait_model.pkl             # Trained machine learning model
│   ├── label_encoder.pkl          # Label encoder for class names
│   ├── gait_dataset.pkl           # Dataset for evaluation
├── app.py                         # Flask application
├── analyze_model_performance.py   # Script for model performance analysis
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
```

## Dependencies
- Python 3.10+
- Flask
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- dataframe-image

Install all dependencies using:
```bash
pip install -r requirements.txt
```

