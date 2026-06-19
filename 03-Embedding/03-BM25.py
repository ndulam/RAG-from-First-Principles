from collections import Counter
import math
# Wukong's battle log
battle_logs = [
    "Wukong,uses,Flaming Fist,to repel,the demon;then activates,Vajra Body,to block,the divine weapon's,attack.",
    "The demon,uses,Frost Arrow,to attack,Wukong,but is,counterattacked and crushed,by,Flaming Fist.",
    "Wukong,summons,Flaming Fist,and,Devastating Roar,to defeat,the demon,then,collects,the demon's,essence."
]
# Hyperparameters
k1 = 1.5
b = 0.75
# Build the vocabulary
vocabulary = set(word for log in battle_logs for word in log.split(","))
vocab_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
# Compute IDF
N = len(battle_logs)
df = Counter(word for log in battle_logs for word in set(log.split(",")))
idf = {word: math.log((N - df[word] + 0.5) / (df[word] + 0.5) + 1) for word in vocabulary}
# Log length info
avg_log_len = sum(len(log.split(",")) for log in battle_logs) / N
# Generate BM25 sparse embeddings
def bm25_sparse_embedding(log):
    tf = Counter(log.split(","))
    log_len = len(log.split(","))
    embedding = {}
    for word, freq in tf.items():
        if word in vocabulary:
            idx = vocab_to_idx[word]
            score = idf[word] * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * log_len / avg_log_len))
            embedding[idx] = score
    return embedding
# Generate the sparse vector
for log in battle_logs:
    sparse_embedding = bm25_sparse_embedding(log)
print(f"Sparse embedding: {sparse_embedding}")
