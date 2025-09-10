from sentence_transformers import SentenceTransformer

model = SentenceTransformer("cross-encoder/nli-deberta-v3-small")

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium."
]
embeddings = model.encode(sentences)

similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]