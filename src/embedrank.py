import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity

from nlp_uitl import extract_keyphrase_candidates, tokenize


class EmbedRank():
    def __init__(self, model, tokenize, N=5, l=0.55):
        self.model = model
        self.tokenize = tokenize
        self.N = N
        self.l = l
        self.phrases = []
        self.phrase_embeddings = []
        self.document_embedding = []
        self.document_similarity = []

    def extract_keyword(self, text):
        phrase_indices = self._mmr(text)
        output = []
        for idx in phrase_indices:
            output.append((self.phrases[idx], self.document_similarity[idx][0]))
        return output

    def _mmr(self, document):
        self.document_embedding = self.model.infer_vector(self.tokenize(document))

        self.phrases = []
        self.phrase_embeddings = []
        for candidate_tokens in extract_keyphrase_candidates(document):
            candidate_text = "".join(candidate_tokens)
            self.phrases.append(candidate_text)
            self.phrase_embeddings.append(self.model.infer_vector(self.tokenize(candidate_text)))
        self.phrase_embeddings = np.array(self.phrase_embeddings)

        if len(self.phrases) == 0:
            return []
        if len(self.phrases) < self.N:
            # Num of candidate phrases are smaller than extracted num:
            #   extract all phrases and reranked by mmr
            self.N = len(self.phrases)

        self.document_similarity = cosine_similarity(self.phrase_embeddings, self.document_embedding.reshape(1, -1))
        phrase_similarity_matrix = cosine_similarity(self.phrase_embeddings)

        # MMR
        # 1st iteration
        unselected = list(range(len(self.phrases)))
        select_idx = np.argmax(self.document_similarity)

        selected = [select_idx]
        unselected.remove(select_idx)

        # other iterations
        for _ in range(self.N - 1):
            mmr_distance_to_doc = self.document_similarity[unselected, :]
            mmr_distance_between_phrases = np.max(phrase_similarity_matrix[unselected][:, selected], axis=1)

            mmr = self.l * mmr_distance_to_doc - (1 - self.l) * mmr_distance_between_phrases.reshape(-1, 1)
            mmr_idx = unselected[np.argmax(mmr)]

            selected.append(mmr_idx)
            unselected.remove(mmr_idx)

        return selected
