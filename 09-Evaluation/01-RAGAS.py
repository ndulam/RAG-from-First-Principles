import os
from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file
import numpy as np
from datasets import Dataset
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate

# Prepare LLM for evaluation (using GPT-3.5)
# Use Ragas's LangchainLLMWrapper to wrap LangChain's ChatOpenAI model
llm = LangchainLLMWrapper(ChatOpenAI(model_name="gpt-3.5-turbo"))

# Prepare dataset
# This dataset contains questions, generated answers, and relevant context information
data = {
    "question": [
        "Who is the main character in Black Myth: Wukong?",
        "What are the special features of the combat system in Black Myth: Wukong?",
        "How is the visual quality of Black Myth: Wukong?",
    ],
    "answer": [
        "The main character in Black Myth: Wukong is Sun Wukong, based on the Chinese classic 'Journey to the West' but with a new interpretation. This version of Sun Wukong is more mature and brooding, showing a different personality from the traditional character.",
        "Black Myth: Wukong's combat system combines Chinese martial arts with Soulslike game features, including light and heavy attack combinations, technique transformations, and magic systems. Notably, Wukong can transform between different weapon forms during combat, such as his iconic staff and nunchucks, and use various mystical abilities.",
        "Black Myth: Wukong is developed using Unreal Engine 5, showcasing stunning visual quality. The game's scene modeling, lighting effects, and character details are all top-tier, particularly in its detailed recreation of traditional Chinese architecture and mythological settings.",
    ],
    "contexts": [
        [
            "Black Myth: Wukong is an action RPG developed by Game Science, featuring Sun Wukong as the protagonist based on 'Journey to the West' but with innovative interpretations. In the game, Wukong has a more composed personality and carries a special mission.",
            "The game is set in a mythological world, telling a new story that presents a different take on the traditional Sun Wukong character."
        ],
        [
            "The game's combat system is heavily influenced by Soulslike games while incorporating traditional Chinese martial arts elements. Players can utilize different weapon forms, including the iconic staff and other transforming weapons.",
            "During combat, players can unleash various mystical abilities, combined with light and heavy attacks and combo systems, creating a fluid and distinctive combat experience. The game also features a unique transformation system."
        ],
        [
            "Black Myth: Wukong demonstrates exceptional visual quality, built with Unreal Engine 5, achieving extremely high graphical fidelity. The game's environments and character models are meticulously crafted.",
            "The lighting effects, material rendering, and environmental details all reach AAA-level standards, perfectly capturing the atmosphere of an Eastern mythological world."
        ]
    ]
}

# Convert the dictionary to a Hugging Face Dataset object for Ragas processing
dataset = Dataset.from_dict(data)

print("\n=== Ragas Evaluation Metrics Description ===")
print("\n1. Faithfulness")
print("- Evaluates whether the generated answer is faithful to the context content")
print("- By breaking down the answer into simple statements and verifying if each statement can be inferred from the context")
print("- This metric only relies on LLM and does not require an embedding model")

# Evaluate Faithfulness
# Create Faithfulness evaluation metric, which only requires an LLM for evaluation
faithfulness_metric = [Faithfulness(llm=llm)] # Only need to provide the generation model
print("\nEvaluating Faithfulness...")
# Use the evaluate function to evaluate the dataset
faithfulness_result = evaluate(dataset, faithfulness_metric)
# Extract faithfulness score
scores = faithfulness_result['faithfulness']
# Calculate average score
mean_score = np.mean(scores) if isinstance(scores, (list, np.ndarray)) else scores
print(f"Faithfulness Score: {mean_score:.4f}")

print("\n2. AnswerRelevancy")
print("- Evaluates the relevance of the generated answer to the question")
print("- Uses an embedding model to calculate semantic similarity")
print("- We will compare open-source embedding models and OpenAI's embedding models")

# Set up two embedding models
# Use Ragas's LangchainEmbeddingsWrapper to wrap LangChain's embedding models
# 1. Open-source all-MiniLM-L6-v2 model
opensource_embedding = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)
# 2. OpenAI's text-embedding-ada-002 model
openai_embedding = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-ada-002"))

# Create AnswerRelevancy evaluation metric
# Create AnswerRelevancy evaluation metrics for both embedding models
opensource_relevancy = [AnswerRelevancy(llm=llm, embeddings=opensource_embedding)]
openai_relevancy = [AnswerRelevancy(llm=llm, embeddings=openai_embedding)]

print("\nEvaluating Answer Relevancy...")
print("\nEvaluating with Open-source Embedding Model:")
# Evaluate using the open-source embedding model
opensource_result = evaluate(dataset, opensource_relevancy)
scores = opensource_result['answer_relevancy']
opensource_mean = np.mean(scores) if isinstance(scores, (list, np.ndarray)) else scores
print(f"Relevancy Score: {opensource_mean:.4f}")

print("\nEvaluating with OpenAI Embedding Model:")
# Evaluate using the OpenAI embedding model
openai_result = evaluate(dataset, openai_relevancy)
scores = openai_result['answer_relevancy']
openai_mean = np.mean(scores) if isinstance(scores, (list, np.ndarray)) else scores
print(f"Relevancy Score: {openai_mean:.4f}")

# Compare results of two embedding models
print("\n=== Embedding Model Comparison ===")
diff = openai_mean - opensource_mean
print(f"Open-source Model Score: {opensource_mean:.4f}")
print(f"OpenAI Model Score: {openai_mean:.4f}")
print(f"Difference: {diff:.4f} ({'OpenAI is better' if diff > 0 else 'Open-source model is better' if diff < 0 else 'Similar'})")


'''
I made the following changes:
Removed HuggingfaceEmbeddings import from ragas.embeddings.base
Changed to import LangChain's HuggingFaceEmbeddings
Used LangchainEmbeddingsWrapper to wrap LangChain's HuggingFaceEmbeddings
The reasons for this are:
LangChain's HuggingFaceEmbeddings is a complete implementation that includes all necessary methods
LangchainEmbeddingsWrapper adapts LangChain's embedding model to the RAGAS interface
This wrapper automatically handles the conversion of synchronous and asynchronous methods
1. Faithfulness
- Evaluates whether the generated answer is faithful to the context content
- By breaking down the answer into simple statements and verifying if each statement can be inferred from the context
- This metric only relies on LLM and does not require an embedding model

Evaluating Faithfulness...
Evaluating: 100%|███████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:05<00:00,  1.87s/it]
Faithfulness Score: 0.6071

Evaluating Answer Relevancy...

Evaluating with Open-source Embedding Model:
Evaluating: 100%|███████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  1.54it/s]
Relevancy Score: 0.8565

Evaluating with OpenAI Embedding Model:
Evaluating: 100%|███████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:06<00:00,  2.11s/it]
Relevancy Score: 0.9426

=== Embedding Model Comparison ===
Open-source Model Score: 0.8565
OpenAI Model Score: 0.9426
Difference: 0.0861 (OpenAI is better)


'''