import sqlite3

def check_sqlite_integrity(db_file):
    con = sqlite3.connect(db_file)
    cursor = con.cursor()
    cursor.execute("PRAGMA integrity_check;")
    result = cursor.fetchone()[0]
    if result.lower() == "ok":
        print("SQLite database integrity check passed.")
    else:
        print("SQLite database integrity check failed:", result)
    con.close()

if __name__ == '__main__':
    db_file = '../results/m4efad.db'
    check_sqlite_integrity(db_file)

