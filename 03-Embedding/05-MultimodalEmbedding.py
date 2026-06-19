"""
Simple multimodal embedding example: encode an image using the Visualized-BGE model
Install visual_bge by following:
https://github.com/FlagOpen/FlagEmbedding/tree/master/research/visual_bge#readme

# If visual_bge still doesn't work after installing it, try switching to a physical
# environment -- it may not work in a virtual environment.
"""

import torch
from visual_bge.modeling import Visualized_BGE
from PIL import Image
import numpy as np

# Initialize the encoder
model_name = "BAAI/bge-base-en-v1.5"
# Define the model path (use an absolute path if you run into issues)
# Download the model weights file beforehand
# wget https://huggingface.co/BAAI/bge-visualized/resolve/main/Visualized_base_en_v1.5.pth
model_path = "/root/AI-BOX/code/rag/rag-in-action/03-Embedding/Visualized_base_en_v1.5.pth"
model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)
model.eval()

# Define the image path (use an absolute path if you run into issues)
image_path = "/root/AI-BOX/code/rag/rag-in-action/90-Data/Multimodal/query_image.jpg"

# Encode the image
with torch.no_grad():
    # Encode using only the image
    image_embedding = model.encode(image=image_path)

    # Encode using both image and text
    text = "This is an example image of Wukong in combat"
    multimodal_embedding = model.encode(image=image_path, text=text)

# Move the tensors to CPU and convert to numpy arrays
image_embedding_np = image_embedding.cpu().numpy()
multimodal_embedding_np = multimodal_embedding.cpu().numpy()

# Print embedding vector information
print("=== Image Embedding Vector Info ===")
print(f"Vector dimension: {image_embedding_np.shape[1]}")
print(f"Vector sample (first 10 elements): {image_embedding_np[0][:10]}")
print(f"Vector norm: {np.linalg.norm(image_embedding_np[0])}")

print("\n=== Multimodal Embedding Vector Info ===")
print(f"Vector dimension: {multimodal_embedding_np.shape[1]}")
print(f"Vector sample (first 10 elements): {multimodal_embedding_np[0][:10]}")
print(f"Vector norm: {np.linalg.norm(multimodal_embedding_np[0])}")