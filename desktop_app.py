from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QIcon  # Import QIcon for setting the app icon
import threading
import sys
from app import app  # Import your actual Flask app

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gait-Based Identification System")  # Set the app name
        self.setGeometry(100, 100, 1024, 768)

        # Set the application icon
        self.setWindowIcon(QIcon("static/images/logo.ico"))  # Path to the .ico file

        # Create a web view to display the Flask app
        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl("http://127.0.0.1:5000"))  # Flask app URL
        self.setCentralWidget(self.browser)

def run_flask():
    """Run the Flask app in a separate thread."""
    app.run(debug=False, use_reloader=False)  # Disable debug mode and reloader for production

if __name__ == "__main__":
    # Start the Flask app in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True  # Ensure the thread exits when the main program exits
    flask_thread.start()

    # Start the PyQt application
    qt_app = QApplication(sys.argv)
    qt_app.setWindowIcon(QIcon("static/images/logo.ico"))  # Set the app icon globally
    window = MainWindow()
    window.show()
    sys.exit(qt_app.exec_())
