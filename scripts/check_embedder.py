"""Quick manual check of the E5 embeddings (run with ``python -m scripts.check_embedder``)."""

from app.search.embedder import E5Embedder


def main():
    print("Chargement du modèle e5...")
    model = E5Embedder()

    docs = [
        "Seigneur, enseigne-nous à prier.",
        "Notre Père qui es aux cieux.",
        "Ne crains point, car je suis avec toi.",
    ]
    query = "Dans quel passage Jésus enseigne-t-il sur la prière ?"

    doc_vecs = model.embed(docs, is_query=False)
    q_vec = model.embed([query], is_query=True)[0]

    scores = doc_vecs @ q_vec
    print("Similarités (plus haut = plus pertinent) :")
    for text, score in sorted(zip(docs, scores), key=lambda x: -x[1]):
        print(f"  {score:.3f}  |  {text}")


if __name__ == "__main__":
    main()
