from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

"""
CrossEncoder reranking algorithm implementation

CrossEncoder is a BERT-based bidirectional encoder reranking model, specifically designed to compute relevance scores for query-document pairs.

Core principle:
1. Feed the query and document into the BERT model together as a single input
2. Use the output at the [CLS] token to predict the relevance score
3. Trained end-to-end, enabling it to capture deep interactions between the query and the document

Differences from other methods:
- Compared to the dual-tower model (Bi-Encoder): CrossEncoder models the interaction between query and document better
- Compared to traditional BM25: it can understand semantic similarity, not just rely on keyword matching
- Compared to simple vector similarity: it accounts for positional information and contextual relationships between the query and document

Advantages:
- High accuracy: can precisely model the relevance of query-document pairs
- Strong semantic understanding: based on a pretrained language model with powerful semantic comprehension
- Good adaptability: can be fine-tuned to adapt to specific domains

Disadvantages:
- High computational cost: requires encoding each query-document pair separately
- Poor real-time performance: not suitable for the first stage of large-scale retrieval, typically used in the reranking stage
"""

print("🔄 Initializing CrossEncoder reranking model...")

# 1. Load the pretrained CrossEncoder model
print("📥 Loading pretrained model...")
model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"  # A small model trained on the MS MARCO dataset
print(f"Using model: {model_name}")
print("  - This model is fine-tuned on the MS MARCO passage retrieval task")
print("  - Specifically optimized to compute query-passage relevance scores")
print("  - Balances model size and performance, suitable for production use")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
print("✅ Model loading complete")

# 2. Prepare test data
print("\n📋 Preparing test data...")
query = "What are some famous tourist attractions in Shanxi?"
documents = [
    "Mount Wutai is one of China's four sacred Buddhist mountains, famous as the bodhimanda of Manjushri Bodhisattva.",
    "Yungang Grottoes is one of China's three great grotto complexes, renowned for its exquisite Buddhist sculptures.",
    "Pingyao Ancient City is one of China's best-preserved ancient county towns, listed as a UNESCO World Heritage Site.",
]

print(f"Query: {query}")
print(f"Number of candidate documents: {len(documents)}")
for i, doc in enumerate(documents, 1):
    print(f"  Document {i}: {doc}")

def encode_and_score(query, docs):
    """
    CrossEncoder relevance scoring function

    Purpose: compute the relevance score between the query and each document

    Args:
        query (str): the user query
        docs (list): list of candidate documents

    Returns:
        list: the relevance score corresponding to each document

    Workflow:
        1. Concatenate the query and document into the "[CLS] query [SEP] document [SEP]" format
        2. Encode via the tokenizer, generating input_ids, attention_mask, etc.
        3. Feed into the BERT model and obtain the output at the [CLS] position
        4. Compute the relevance score through the classification head
        5. A higher score indicates stronger relevance
    """
    print(f"\n🧠 Computing relevance scores for {len(docs)} documents...")
    scores = []

    for i, doc in enumerate(docs, 1):
        print(f"  Processing document {i}/{len(docs)}...")

        # Combine the query and document into BERT input format
        # Format: [CLS] query [SEP] document [SEP]
        inputs = tokenizer(
            query,
            doc,
            return_tensors="pt",           # Return PyTorch tensors
            truncation=True,               # Truncate overly long inputs
            max_length=512,                # Maximum BERT input length
            padding="max_length"           # Pad to the maximum length
        )

        # Forward pass to compute the relevance score
        with torch.no_grad():  # Disable gradient computation to save memory
            outputs = model(**inputs)
            # Get the logits (raw scores); the first element is usually the relevance score
            score = outputs.logits[0][0].item()
            scores.append(score)

        print(f"    Query-document pair relevance score: {score:.4f}")
        print(f"    Input length: {len(inputs['input_ids'][0])} tokens")

    print("✅ Relevance score computation complete")
    return scores

# 3. Run CrossEncoder reranking
print(f"\n🎯 Running CrossEncoder reranking...")
scores = encode_and_score(query, documents)

# 4. Sort documents by score
print(f"\n📊 Sorting documents by relevance score...")
ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

# 5. Output the reranking results
print(f"\n{'='*60}")
print(f"🏆 CrossEncoder Reranking Results")
print(f"{'='*60}")
print(f"Query: {query}")
print(f"\nRanking results (sorted by relevance score, descending):")

for rank, (doc, score) in enumerate(ranked_docs, start=1):
    print(f"\n📄 Rank {rank}:")
    print(f"   Relevance score: {score:.4f}")
    print(f"   Document content: {doc}")

    # Interpret the meaning of the score
    if score > 0:
        relevance_level = "Highly relevant"
    elif score > -2:
        relevance_level = "Moderately relevant"
    else:
        relevance_level = "Low relevance"
    print(f"   Relevance level: {relevance_level}")

print(f"\n📋 CrossEncoder Reranking Summary:")
print("- ✅ Deep semantic understanding: captures fine-grained interactions between query and document")
print("- ✅ Precise relevance modeling: end-to-end training yields accurate relevance scores")
print("- ✅ Context-aware: accounts for positional information and contextual relationships of tokens")
print("- ⚠️  Computationally intensive: each query-document pair must be encoded separately")
print("- 💡 Best practice: use to finely rerank initial retrieval results")
