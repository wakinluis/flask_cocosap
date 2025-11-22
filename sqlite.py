import sqlite3

conn = sqlite3.connect("ispindel.db")
cur = conn.cursor()

def execute(query):
    try:
        cur.execute(query)
    except sqlite3.OperationalError as e:
        print(f"Error creating table: {e}")

# Batches table
execute("""
    CREATE TABLE IF NOT EXISTS batches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id TEXT UNIQUE,
        start_date TEXT,
        end_date TEXT,
        liter REAL,
        is_logging INTEGER DEFAULT 0,
        fermentation_status INTEGER DEFAULT 0,
        prediction_value REAL
    )
    """)

# Readings table
execute("""
    CREATE TABLE IF NOT EXISTS readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id TEXT NOT NULL,
        gravity REAL,
        temperature REAL,
        battery REAL,
        angle REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        brix REAL GENERATED ALWAYS AS (((182.4601 * gravity - 775.6821) * gravity + 1262.7794) * gravity - 669.5622) STORED
    ) 
    """)

# ABV table
execute("""
    CREATE TABLE IF NOT EXISTS abv (
        batch_id TEXT NOT NULL,
        original_gravity REAL,
        final_gravity REAL,
        estimated_abv REAL,
        current_abv REAL
    )
    """)

conn.commit()
conn.close()
print("Database and tables created successfully.")