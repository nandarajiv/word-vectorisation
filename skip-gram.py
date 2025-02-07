import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pickle

torch.manual_seed(0)

def get_sentences(file_name):
    with open(file_name, "r") as f:
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


def get_vocab(sentences):
    freq = {}
    for sentence in tqdm(sentences, desc="Calculating word frequencies"):
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
    for i in tqdm(range(len(sentences)), desc="Building vocab"):
        for j in range(len(sentences[i])):
            if sentences[i][j] in rare_words:
                sentences[i][j] = "<unk>"
            if sentences[i][j] not in vocab:
                vocab[sentences[i][j]] = len(vocab)

    return vocab


def get_idx2word(vocab):
    return {v: k for k, v in vocab.items()}


def get_positive_samples(sentences, vocab):
    positive_samples = []
    positive_samples_dict = {}
    for sentence in tqdm(sentences, desc="Getting positive samples"):
        for i in range(len(sentence)):
            for j in range(max(0, i - 8), min(len(sentence), i + 9)):
                if i != j:
                    positive_samples.append((vocab[sentence[i]], vocab[sentence[j]]))
                    positive_samples_dict[(vocab[sentence[i]], vocab[sentence[j]])] = 1

    return positive_samples, positive_samples_dict


def get_negative_samples(vocab, positive_samples):
    negative_samples = []
    vocab_len = len(vocab)
    for i in tqdm(range(3 * len(positive_samples)), desc="Getting negative samples"):
        word1 = np.random.randint(0, vocab_len)
        word2 = np.random.randint(0, vocab_len)
        if word1 != word2 and word1 != 1 and word2 != 1:
            negative_samples.append((word1, word2))

    return negative_samples


def remove_duplicate_negative_samples(positive_samples_dict, negative_samples):
    negative_samples_dict = {}
    count_retained = 0
    for sample in tqdm(negative_samples, desc="Removing duplicate negative samples"):
        if sample not in positive_samples_dict:
            negative_samples_dict[sample] = 1
            count_retained += 1

    return list(negative_samples_dict.keys()), count_retained


def get_samples(positive_samples, negative_samples):
    samples = []
    for sample in tqdm(positive_samples, desc="Adding positive samples"):
        samples.append((sample, 1))
    for sample in tqdm(negative_samples, desc="Adding negative samples"):
        samples.append((sample, 0))

    return samples


class Word2VecDataset(Dataset):
    def __init__(self, samples, vocab):
        self.samples = samples
        self.vocab = vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(sample[0][0]),
            torch.tensor(sample[0][1]),
            torch.tensor(sample[1]),
        )


class SkipGramBinaryClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramBinaryClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(2 * embedding_dim, 324)
        self.non_linear_layer = nn.ReLU()
        self.fc2 = nn.Linear(324, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, target, context):
        target_embedding = self.embedding(target)
        context_embedding = self.embedding(context)

        x = torch.cat((target_embedding, context_embedding), dim=1)
        x = self.fc1(x)
        x = self.non_linear_layer(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x


def train_model(model, train_loader, num_epochs, optimizer, loss_fn, device, accuracies):
    model.to(device)
    model.eval()
    total = 0
    correct = 0
    for word1, word2, label in tqdm(train_loader, desc="Pre evaluation"):
        word1 = word1.to(device)
        word2 = word2.to(device)
        label = label.to(device)

        output = model(word1, word2)
        output = torch.round(output)
        correct += (output == label.unsqueeze(1)).sum().item()
        total += len(label)

    print(" Accuracy: ", correct / total)
    accuracies.append(correct / total)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for word1, word2, label in tqdm(
            train_loader, desc="Training epoch " + str(epoch + 1)
        ):
            word1 = word1.to(device)
            word2 = word2.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(word1, word2)
            loss = loss_fn(output, label.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("Epoch ", epoch, " Loss: ", total_loss)

        # evaluate_model(model, val_loader, loss_fn, device)
        model.eval()
        total = 0
        correct = 0
        for word1, word2, label in tqdm(
            train_loader, desc="Evaluating epoch " + str(epoch + 1)
        ):
            word1 = word1.to(device)
            word2 = word2.to(device)
            label = label.to(device)

            output = model(word1, word2)
            output = torch.round(output)
            correct += (output == label.unsqueeze(1)).sum().item()
            total += len(label)

        print("Epoch ", epoch, " Accuracy: ", correct / total)
        accuracies.append(correct / total)


def get_embeddings(model):
    embeddings = model.embedding.weight.detach().cpu().numpy()
    return embeddings


def save_embeddings(embeddings):
    # save as pt
    em_tensor = torch.tensor(embeddings)
    torch.save(em_tensor, "skip-gram-word-vectors.pt")


def load_embeddings():
    embeddings_tensor = torch.load("skip-gram-word-vectors.pt")
    return embeddings_tensor.numpy()


def save_vocab(vocab):
    with open("skip-gram-vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)


def load_vocab():
    with open("skip-gram-vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    return vocab

def get_k_closest_words(embeddings, vocab, idx2word, word, k=10):
    word_idx = vocab[word]
    word_vec = embeddings[word_idx]
    dist = np.linalg.norm(embeddings - word_vec, axis=1)
    closest_words = np.argsort(dist)
    return [idx2word[i] for i in closest_words[:k]]


if __name__ == "__main__":
    sentences = get_sentences("ANLP-2/train.csv")

    vocab = get_vocab(sentences)
    idx2word = get_idx2word(vocab)

    positive_samples, positive_samples_dict = get_positive_samples(sentences, vocab)
    negative_samples = get_negative_samples(vocab, positive_samples)

    negative_samples, count_retained = remove_duplicate_negative_samples(
        positive_samples_dict, negative_samples
    )

    samples = get_samples(positive_samples, negative_samples)

    skipgram_dataset = Word2VecDataset(samples, vocab)

    train_loader = DataLoader(skipgram_dataset, batch_size=512, shuffle=True)

    model = SkipGramBinaryClassifier(len(vocab), 300)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    accuracies = []
    train_model(
        model,
        train_loader,
        5,
        torch.optim.Adam(model.parameters()),
        nn.BCELoss(),
        device,
        accuracies,
    )

    embeddings = get_embeddings(model)

    save_embeddings(embeddings)
    save_vocab(vocab)

    print(get_k_closest_words(embeddings, vocab, idx2word, 'vaccine'))
    print("accuracies: ")
    print(accuracies)
