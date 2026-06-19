# Connect to the SQLite database
import sqlite3
conn = sqlite3.connect('90-Data/tourism.db')
cursor = conn.cursor()
# Create the scenic spots table
cursor.execute('''
CREATE TABLE IF NOT EXISTS scenic_spots (
    scenic_id INTEGER PRIMARY KEY,
    scenic_name VARCHAR(100) NOT NULL,
    city VARCHAR(50) NOT NULL,
    level VARCHAR(20),
    monthly_visitors INTEGER
)''')
# Create the city info table
cursor.execute('''
CREATE TABLE IF NOT EXISTS city_info (
    city_id INTEGER PRIMARY KEY,
    city_name VARCHAR(50) NOT NULL,
    annual_tourism_income INTEGER,
    famous_dish VARCHAR(100)
)''')
# Insert sample data into the scenic spots table
sample_scenic_spots = [
    (1, 'Jinci Temple', 'Taiyuan', 'AAAAA', 50000),
    (2, 'Mount Wutai', 'Xinzhou', 'AAAAA', 80000),
    (3, 'Yungang Grottoes', 'Datong', 'AAAAA', 70000),
    (4, 'Pingyao Ancient City', 'Jinzhong', 'AAAAA', 90000),
    (5, 'Qiao Family Compound', 'Jinzhong', 'AAAA', 45000)
]
cursor.executemany('INSERT OR REPLACE INTO scenic_spots VALUES (?, ?, ?, ?, ?)', sample_scenic_spots)
# Insert sample data into the city info table
sample_city_info = [
    (1, 'Taiyuan', 200000000, 'Daoxiao noodles'),
    (2, 'Datong', 180000000, 'Datong vinegar'),
    (3, 'Jinzhong', 150000000, 'Saozi noodles'),
    (4, 'Xinzhou', 120000000, 'Youmian kaolaolao'),
    (5, 'Yuncheng', 130000000, 'Yuncheng boiled cakes')
]
cursor.executemany('INSERT OR REPLACE INTO city_info VALUES (?, ?, ?, ?)', sample_city_info)
# Commit the changes and close the connection
conn.commit()
conn.close()
print("Database tables created and sample data inserted.")
