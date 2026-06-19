from FlagEmbedding import BGEM3FlagModel

def main():
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    passage = ["Wukong used Flaming Fist to repel the demon, then activated Vajra Body to block the divine weapon's attack."]

    # Encode the text and get sparse and dense embeddings
    passage_embeddings = model.encode(
        passage,
        return_sparse=True,     # return sparse embeddings
        return_dense=True,      # return dense embeddings
        return_colbert_vecs=True  # return multi-vector embeddings
    )
    # Extract the sparse, dense, and multi-vector embeddings
    dense_vecs = passage_embeddings["dense_vecs"]
    sparse_vecs = passage_embeddings["lexical_weights"]
    colbert_vecs = passage_embeddings["colbert_vecs"]
    # Show examples of the sparse and dense embeddings
    print("Dense embedding dimension:", dense_vecs[0].shape)
    print("Dense embedding, first 10 dims:", dense_vecs[0][:10])  # only show the first 10 dims

    print("Sparse embedding total length:", len(sparse_vecs[0]))
    print("Sparse embedding, first 10 non-zero values:", list(sparse_vecs[0].items())[:10])  # only show the first 10 non-zero values

    print("Multi-vector embedding dimension:", colbert_vecs[0].shape)
    print("Multi-vector embedding, first 2:", colbert_vecs[0][:2])  # only show the first 2 multi-vector embeddings

if __name__ == '__main__':
    main()
