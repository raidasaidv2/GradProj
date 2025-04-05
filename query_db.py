import sqlite3

def query_users():
    conn = sqlite3.connect('gait_users.db')
    cursor = conn.cursor()

    # Query the users table
    cursor.execute("SELECT * FROM users")
    rows = cursor.fetchall()

    print("Users Table:")
    for row in rows:
        print(row)

    conn.close()

def query_employees():
    conn = sqlite3.connect('gait_users.db')
    cursor = conn.cursor()

    # Query the employees table
    cursor.execute("SELECT * FROM employees")
    rows = cursor.fetchall()

    print("\nEmployees Table:")
    for row in rows:
        print(row)

    conn.close()

if __name__ == '__main__':
    query_users()
    query_employees()
