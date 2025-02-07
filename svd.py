import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import torch
import pickle


class SVD_embedder:
    def __init__(self, file_name, embedding_dim):
        self.file_name = file_name
        self.embedding_dim = embedding_dim
        self.sentences = None
        self.vocab = None
        self.comatrix = None
        self.idx2word = None
        self.embeddings = self.get_embeddings()
        self.save_embeddings()
        self.save_vocab()

    def get_sentences(self):
        with open(self.file_name, "r") as f:
            raw_sentences = f.readlines()

        count = 0
        sentences = []

        for line in raw_sentences:
            if count == 0:
                count += 1
                continue
            if count == 15001:
                break

            temp = line.split(",")[1:]
            temp = "".join(temp)
            temp = temp.lower()
            temp = temp.replace("\\", " ")
            temp = temp.replace("$", "dollar ")
            temp = "".join(e for e in temp if e.isalnum() or e.isspace())

            temp = word_tokenize(temp)

            temp = [
                word if not any(char.isdigit() for char in word) else "<num>"
                for word in temp
            ]
            temp = ["<s>"] + temp + ["</s>"]

            sentences.append(temp)
            count += 1

        return sentences

    def get_vocab(self):
        freq = {}
        for sentence in tqdm(self.sentences, desc="Calculating word frequencies"):
            for word in sentence:
                if word in freq:
                    freq[word] += 1
                else:
                    freq[word] = 1

        rare_words = [word for word in freq if freq[word] < 4]

        vocab = {}
        vocab["<unk>"] = 0
        vocab["<pad>"] = 1
        vocab["<s>"] = 2
        vocab["</s>"] = 3
        for i in tqdm(range(len(self.sentences)), desc="Building vocab"):
            for j in range(len(self.sentences[i])):
                if self.sentences[i][j] in rare_words:
                    self.sentences[i][j] = "<unk>"
                if self.sentences[i][j] not in vocab:
                    vocab[self.sentences[i][j]] = len(vocab)

        return vocab

    def get_idx2word(self):
        return {v: k for k, v in self.vocab.items()}

    def build_co_matrix(self):
        matrix = np.zeros((len(self.vocab), len(self.vocab)))
        for sentence in tqdm(self.sentences, desc="Building co-occurrence matrix"):
            for i in range(len(sentence)):
                for j in range(max(0, i - 9), min(len(sentence), i + 10)):
                    if i != j:
                        matrix[self.vocab[sentence[i]], self.vocab[sentence[j]]] += 1

        return matrix

    def get_embeddings(self):
        self.sentences = self.get_sentences()
        self.vocab = self.get_vocab()
        self.idx2word = self.get_idx2word()
        self.comatrix = self.build_co_matrix()

        print("Performing SVD, this may take a while...")
        U, S, V = np.linalg.svd(self.comatrix)

        return U[:, : self.embedding_dim]

    def save_vocab(self):
        with open("svd-vocab.pkl", "wb") as f:
            pickle.dump(self.vocab, f)

    def save_embeddings(self):
        em_tensor = torch.tensor(self.embeddings)
        torch.save(em_tensor, "svd-word-vectors.pt")

    def load_embeddings(self):
        embeddings_tensor = torch.load("svd-word-vectors.pt")
        return embeddings_tensor.numpy()

    def get_k_closest_words(self, word, k=10):
        word_idx = self.vocab[word]
        word_vec = self.embeddings[word_idx]
        dist = np.linalg.norm(self.embeddings - word_vec, axis=1)
        closest_words = np.argsort(dist)
        return [self.idx2word[i] for i in closest_words[:k]]


if __name__ == "__main__":
    embedder = SVD_embedder("ANLP-2/train.csv", 300)
    print(embedder.get_k_closest_words("vaccine"))
