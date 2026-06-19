# Homework: Students can try to generate an image using MultimodalRAG based on this code framework.
# Not only implement Multimodal retrieval, but also further combine all information based on the retrieved content, and use modern LLMs to generate new text or images.

import os
import weaviate
import weaviate.classes as wvc
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# 1. Connect to Weaviate instance (local or cloud)
client = weaviate.connect_to_local()  # Replace with connect_to_wcs/wcs_cloud if using cloud service

# 2. Create Multimodal Collection
def create_multimodal_collection():
    client.collections.create(
        name="Animals",
        vectorizer_config=wvc.config.Configure.Vectorizer.multi2vec_bind(
            audio_fields=["audio"],
            image_fields=["image"],
            video_fields=["video"],
        )
    )
    print("Multimodal collection 'Animals' created")

# 3. Insert Multimodal Data (taking image as an example)
def insert_multimodal_data():
    animals = client.collections.get("Animals")
    # Assuming there is a base64 string of an image here
    image_base64 = "<YOUR_IMAGE_BASE64_STRING>"
    animals.data.insert({
        "name": "puppy",
        "image": image_base64,
        "mediaType": "image"
    })
    print("Image data inserted")

# 4. Retrieve Image (using text as query)
def retrieve_image(query):
    animals = client.collections.get("Animals")
    response = animals.query.near_text(
        query=query,
        filters=wvc.query.Filter(path="mediaType").equal("image"),
        return_properties=['name','mediaType','image'],
        limit=1,
    )
    result = response.objects[0].properties
    print("Retrieved image object:", result)
    return result['image']

# 5. Generate Image Description with GPT-4V
def generate_description_from_image_gpt4(prompt, image64, openai_api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image64}"}}
                ]
            }
        ],
        "max_tokens": 300
    }
    response_oai = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    result = response_oai.json()['choices'][0]['message']['content']
    print(f"Generated description: {result}")
    return result

# 6. Generate Image with DALL-E-3 based on description
def generate_image_dalee3(prompt, openai_api_key):
    openai_client = OpenAI(api_key=openai_api_key)
    response_oai = openai_client.images.generate(
        model="dall-e-3",
        prompt=str(prompt),
        size="1024x1024",
        quality="standard",
        n=1,
    )
    result = response_oai.data[0].url
    print(f"Generated image URL: {result}")
    return result

if __name__ == "__main__":
    # Step-by-step demonstration
    # 1. Create collection
    create_multimodal_collection()
    # 2. Insert data (please replace image base64 string first)
    insert_multimodal_data()
    # 3. Retrieve image
    image64 = retrieve_image("dog with a sign")
    # 4. Generate description with GPT-4V (please replace with your OpenAI API Key)
    description = generate_description_from_image_gpt4(
        prompt="This is a picture of my pet, please provide a cute and vivid description.",
        image64=image64,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    # 5. Generate image with DALL-E-3
    generate_image_dalee3(description, openai_api_key=os.getenv("OPENAI_API_KEY"))