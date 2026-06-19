# 1. Prepare the document data
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

docs = [
    "Combat in Black Myth: Wukong feels like a wuxia novel come to life - when the golden cudgel clashes with demons, sparks fly and the moves flow like water. Wukong can switch freely between ferocious and nimble fighting styles, sweeping through enemy ranks with a single strike or darting about like a butterfly among flowers.",
    "The 72 Transformations are more than just shapeshifting - they're a key that unlocks new worlds. Turning into a flying squirrel lets you sneak into demon lairs to gather intel, while becoming a goldfish lets you explore the secrets of deep-sea ruins; every transformation is its own unique adventure.",
    "Every boss fight is a heart-pounding showdown - whether battling a massive nine-headed serpent atop a waterfall, or trading spells with the Thunder God and Lightning Mother amid a storm-wracked sea of clouds, every move carries real danger.",
    "Soaring across this mythic world on the Cloud-Somersault, the breathtaking scenery takes your breath away. Mist-wreathed immortal mountains drift in and out of view, ancient monster lairs hide treasures a thousand years old, and the bell of an old temple echoes through moonlit valleys.",
    "This isn't the Journey to the West you know. As Wukong sets out to uncover the mystery of his origins, he'll encounter gods and demons of every kind - some old acquaintances, like the equally headstrong Nezha; others formidable rivals, like Erlang Shen with his three-pointed double-edged blade.",
    "As the Great Sage Equal to Heaven, Wukong's powers go well beyond the golden cudgel. His Fiery Golden Eyes can see through any demon's disguise, and a single somersault carries him a hundred thousand leagues. These abilities can be upgraded further by collecting materials like meteoric iron and enlightenment stones.",
    "Every corner of the world hides a story. You might discover the ruins of an ancient power deep in a mountain cave, find the treasure vault of long-gone celestial soldiers in a sky palace, or stumble upon a fox spirit selling ginseng fruit at a mortal marketplace.",
    "The story is set in a wild world that predates the Tang dynasty, when the Heavenly Court had not yet established its rule over the Three Realms and demon kings carved up the land among themselves. It was a turbulent age of warring gods and demons vying for supremacy - and the starting point of Wukong's search for the truth.",
    "The game's music is like an epic spanning a thousand years. Guqin and orchestral strings weave together the intensity of battle, while flutes and wooden fish carve out a serene, Zen-like calm. And when Wukong enters key story scenes, the classical-style score makes you feel transported back to that mythic age."
    ]

# 2. Set up the embedding model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
doc_embeddings = model.encode(docs)
print(f"Document embedding dimensions: {doc_embeddings.shape}")

# 3. Create the vector store
import faiss # pip install faiss-cpu
import numpy as np
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings.astype('float32'))
print(f"Number of documents in the vector database: {index.ntotal}")

# 4. Perform similarity search
question = "What are the characteristics of the combat system in Black Myth: Wukong?"
query_embedding = model.encode([question])[0]
distances, indices = index.search(
    np.array([query_embedding]).astype('float32'),
    k=3
)
context = [docs[idx] for idx in indices[0]]
print("\nRetrieved relevant documents:")
for i, doc in enumerate(context, 1):
    print(f"[{i}] {doc}")

# 5. Build the prompt
prompt = f"""Answer the question based on the reference information below, and cite the source numbers.
If the answer cannot be found in the reference information, say that you cannot answer.
Reference information:
{chr(10).join(f"[{i+1}] {doc}" for i, doc in enumerate(context))}
Question: {question}
Answer:"""

# 6. Use DeepSeek to generate the answer
from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{
        "role": "user",
        "content": prompt
    }],
    max_tokens=1024
)
print(f"\nGenerated answer: {response.choices[0].message.content}")
