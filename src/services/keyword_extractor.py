from sentence_transformers import (
    SentenceTransformer,
    util
)


class KeywordExtractor:

    def __init__(
        self,
        model_name="all-MiniLM-L6-v2"
    ):

        self.model = (
            SentenceTransformer(
                model_name
            )
        )

    def extract_keywords(
        self,
        text,
        candidate_terms,
        top_n=5
    ):

        text_embedding = (
            self.model.encode(
                text,
                convert_to_tensor=True
            )
        )

        term_embeddings = (
            self.model.encode(
                candidate_terms,
                convert_to_tensor=True
            )
        )

        similarities = (
            util.pytorch_cos_sim(
                text_embedding,
                term_embeddings
            )[0]
        )

        ranked_terms = sorted(
            zip(
                candidate_terms,
                similarities.tolist()
            ),
            key=lambda item: item[1],
            reverse=True
        )

        return [
            term
            for term, _
            in ranked_terms[:top_n]
        ]