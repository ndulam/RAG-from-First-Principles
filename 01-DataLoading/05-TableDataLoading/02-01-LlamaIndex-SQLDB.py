from llama_index.readers.database import DatabaseReader

# Database setup and table schema:
# 1. Create the database: CREATE DATABASE example_db;
# 2. Use the database: USE example_db;
# 3. Create the Black Myth: Wukong game scenes table:
#    CREATE TABLE game_scenes (
#      id INT AUTO_INCREMENT PRIMARY KEY,
#      scene_name VARCHAR(100) NOT NULL,
#      description TEXT,
#      difficulty_level INT,
#      boss_name VARCHAR(100),
#      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#    );
# 4. Insert Black Myth: Wukong game scene data:
#    INSERT INTO game_scenes (scene_name, description, difficulty_level, boss_name)
#    VALUES
#      ('Flower Fruit Mountain', 'Wukong''s birthplace - lush mountains and rivers, suffused with immortal energy', 2, 'Six-Eared Macaque'),
#      ('Water Curtain Cave', 'A cave within Flower Fruit Mountain, Wukong''s home', 1, NULL),
#      ('Flaming Mountain', 'An unbearably scorching volcanic region, filled with magma and roaring flames', 4, 'Bull Demon King'),
#      ('Dragon Palace', 'The palace of the East Sea Dragon King, a wondrous underwater scene', 3, 'Ao Guang'),
#      ('Spirit Mountain', 'The sacred dwelling of the Buddha, bathed in Buddha-light', 5, 'Tathagata Buddha');

reader = DatabaseReader(
    scheme="mysql",
    host="localhost",
    port=3306,
    user="newuser",
    password="password",
    dbname="example_db"
)

query = "SELECT * FROM game_scenes" # select all game scenes -> Text-to-SQL
documents = reader.load_data(query=query)

print(f"Number of documents loaded from the database: {len(documents)}")
print(documents)

# Reference
# https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo/
