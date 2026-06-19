from llama_index.core import Document

# Create several Document objects and attach metadata to them
documents = [
    Document(
        text="An underground cavern thick with the smell of flame and sulfur, where fire continuously erupts from below, lighting up the entire abyss. Rivers of lava wind through the scene, and burning volcanic rocks float in the air. Wukong must use his jumping ability and golden cudgel to navigate between the lava flows while fighting off fire demons from hell.",
        metadata={
            "filename": "fire_lit_abyss_scene.md",
            "category": "Game scene",
            "file_path": "/data/BlackMythWukong/fire_lit_abyss_scene.md",
            "author": "GameScience",
            "creation_date": "2024-11-20",
            "last_modified_date": "2024-11-21",
            "file_type": "markdown",
            "word_count": 28,
        },
    ),
    Document(
        text="A mountain range so tall it pierces the clouds, wreathed in mist with powerful winds. Wukong must leap across cliffs, fly using the Cloud-Somersault, and keep his balance in the strong wind to make it through. The enemies here are mainly bird demons hidden in the clouds and mechanical rock beasts.",
        metadata={
            "filename": "wind_rising_sky_scene.md",
            "category": "Game scene",
            "file_path": "/data/BlackMythWukong/wind_rising_sky_scene.md",
            "author": "GameScience",
            "creation_date": "2024-11-20",
            "last_modified_date": "2024-11-21",
            "file_type": "markdown",
            "word_count": 28,
        },
    )]

# Print the metadata for each document
for doc in documents:
    print(f"Metadata for {doc.metadata['filename']}:")
    for key, value in doc.metadata.items():
        print(f"  {key}: {value}")
print("-" * 40)
