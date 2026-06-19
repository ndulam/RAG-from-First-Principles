from transformers import AutoTokenizer, AutoModel
import torch

"""
ColBERT (Contextualized Late Interaction over BERT) reranking algorithm implementation

ColBERT is an innovative architecture that combines BERT's deep semantic understanding
with efficient retrieval, using a "late interaction" mechanism.

Core innovations:
1. Early separate encoding: queries and documents are encoded independently, allowing document embeddings to be precomputed
2. Late fine-grained interaction: fine-grained token-level interaction is performed in vector space
3. MaxSim operation: for each token in the query, find the most similar token in the document for matching

Technical advantages:
- High efficiency: documents can be precomputed and indexed, only the query needs to be encoded at query time
- High precision: preserves token-level fine-grained interaction without losing semantic information
- Scalable: supports efficient retrieval over large-scale document collections

Comparison with other methods:
- vs CrossEncoder: faster, supports precomputed document embeddings
- vs Bi-Encoder: more fine-grained interaction, considers token-level matching
- vs traditional methods: stronger semantic understanding, supports fuzzy matching

Use cases:
- First stage of large-scale document retrieval
- Applications that need to balance precision and efficiency
- Latency-sensitive real-time retrieval systems
"""

print("🔄 Initializing ColBERT reranking model...")

# 1. Load BERT model and tokenizer
print("📥 Loading base BERT model...")
model_name = "bert-base-uncased"  # Base BERT model, can be replaced with a model fine-tuned specifically for ColBERT
print(f"Using model: {model_name}")
print("  Note: in practical applications, it is recommended to use a model fine-tuned specifically for ColBERT")
print("  For example: 'colbert-ir/colbertv2.0' or another ColBERT-optimized model")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
print("✅ Model loading complete")

# 2. Prepare test data
print("\n📋 Preparing test data...")
query = "What are some famous tourist attractions in Shanxi?"
documents = [
    "Mount Wutai is one of China's four great Buddhist mountains, famous as the bodhimanda of Manjushri Bodhisattva.",
    "Yungang Grottoes is one of China's three great grottoes, renowned for its exquisite Buddhist sculptures.",
    "Pingyao Ancient City is one of China's best-preserved ancient county towns, listed as a UNESCO World Heritage Site."
]

print(f"Query: {query}")
print(f"Number of candidate documents: {len(documents)}")
for i, doc in enumerate(documents, 1):
    print(f"  Document {i}: {doc}")

def encode_text(texts, max_length=128):
    """
    ColBERT text encoding function

    Purpose: encode text into contextualized token embeddings

    Args:
        texts (list): list of texts to encode
        max_length (int): maximum sequence length

    Returns:
        torch.Tensor: embedding tensor with shape [batch_size, seq_len, hidden_size]

    ColBERT encoding characteristics:
        1. Retains independent embeddings for all tokens (not just [CLS])
        2. Every token has full contextual information
        3. Prepares for later token-level interaction
    """
    print(f"  🔤 Encoding text, max sequence length: {max_length}")

    inputs = tokenizer(
        texts,
        return_tensors="pt",      # Return PyTorch tensors
        padding=True,             # Pad to a uniform length
        truncation=True,          # Truncate overly long sequences
        max_length=max_length
    )

    print(f"    Input shape: {inputs['input_ids'].shape}")

    with torch.no_grad():
        outputs = model(**inputs)

    # Return the hidden states of all tokens (not just [CLS])
    embeddings = outputs.last_hidden_state
    print(f"    Output embedding shape: {embeddings.shape}")

    return embeddings

print(f"\n🧠 Starting ColBERT encoding process...")

# 3. Encode the query and documents separately
print(f"\n1️⃣ Encoding query...")
query_embeddings = encode_text([query])  # [1, seq_len, hidden_size]

print(f"\n2️⃣ Encoding documents...")
doc_embeddings = encode_text(documents)  # [num_docs, seq_len, hidden_size]

def calculate_similarity(query_emb, doc_embs):
    """
    ColBERT similarity calculation function (simplified version)

    Purpose: compute ColBERT similarity scores between the query and documents

    Args:
        query_emb (torch.Tensor): query embedding [1, seq_len, hidden_size]
        doc_embs (torch.Tensor): document embeddings [num_docs, seq_len, hidden_size]

    Returns:
        list: similarity score for each document

    ColBERT similarity calculation steps:
        1. For each query token, find the maximum similarity (MaxSim) against all document tokens
        2. Sum the MaxSim scores of all query tokens to get the final score

    Note: a simplified version (mean pooling) is used here; full ColBERT uses the MaxSim operation
    """
    print(f"\n3️⃣ Computing ColBERT similarity...")
    print("  Note: this is a simplified implementation of ColBERT; the full version uses the MaxSim operation")

    # Simplified version: use mean pooling instead of full token-level interaction
    # In actual ColBERT, fine-grained MaxSim computation would happen here
    query_emb_pooled = query_emb.mean(dim=1)  # [1, hidden_size]
    doc_embs_pooled = doc_embs.mean(dim=1)    # [num_docs, hidden_size]

    print(f"    Query pooled shape: {query_emb_pooled.shape}")
    print(f"    Document pooled shape: {doc_embs_pooled.shape}")

    # L2 normalization, to ensure cosine similarity is computed
    query_emb_norm = query_emb_pooled / query_emb_pooled.norm(dim=1, keepdim=True)
    doc_embs_norm = doc_embs_pooled / doc_embs_pooled.norm(dim=1, keepdim=True)

    # Compute cosine similarity
    scores = torch.mm(query_emb_norm, doc_embs_norm.t())  # [1, num_docs]

    print(f"    Similarity score shape: {scores.shape}")

    return scores.squeeze().tolist()

# 4. Compute similarity scores
scores = calculate_similarity(query_embeddings, doc_embeddings)

# 5. Rank documents
print(f"\n📊 Ranking documents by ColBERT similarity score...")
ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

# 6. Output ranking results
print(f"\n{'='*60}")
print(f"🏆 ColBERT Reranking Results")
print(f"{'='*60}")
print(f"Query: {query}")
print(f"\nRanking results (sorted by similarity score, descending):")

for rank, (doc, score) in enumerate(ranked_docs, start=1):
    print(f"\n📄 Rank {rank}:")
    print(f"   ColBERT similarity score: {score:.4f}")
    print(f"   Document content: {doc}")

    # Explain what the score means
    if score > 0.8:
        relevance_level = "Highly relevant"
    elif score > 0.6:
        relevance_level = "Moderately relevant"
    else:
        relevance_level = "Low relevance"
    print(f"   Relevance level: {relevance_level}")

print(f"\n📋 ColBERT Algorithm Summary:")
print("- ✅ Efficient retrieval: supports document precomputation, low latency at query time")
print("- ✅ Fine-grained interaction: preserves token-level semantic interaction information")
print("- ✅ Scalability: suitable for retrieval over large-scale document collections")
print("- ✅ Balanced performance: achieves a good balance between precision and efficiency")
print("- 💡 Full implementation: it is recommended to use a model fine-tuned specifically for ColBERT")
print("- 🔧 Optimization tip: in practical applications, use the MaxSim operation instead of simplified mean pooling")