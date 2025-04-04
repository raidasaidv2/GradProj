import sqlite3
import bcrypt

def create_database():
    conn = sqlite3.connect('gait_users.db')
    cursor = conn.cursor()

    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')

    conn.commit()
    conn.close()

def add_user(username, password):
    conn = sqlite3.connect('gait_users.db')
    cursor = conn.cursor()

    # Hash the password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # Insert the user into the database
    cursor.execute('''
        INSERT INTO users (username, password)
        VALUES (?, ?)
    ''', (username, hashed_password.decode('utf-8')))

    conn.commit()
    conn.close()

if __name__ == '__main__':
    create_database()
    # Example usage: add a user
    add_user('testuser', 'testpassword')