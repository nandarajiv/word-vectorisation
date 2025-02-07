from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(0)


def get_vocab():
    with open("skip-gram-vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    return vocab


def get_data(file_name):
    with open(file_name, "r") as f:
        raw_sentences = f.readlines()

    count = 0
    data = []

    for line in tqdm(raw_sentences, desc="Reading data"):
        if count == 0:
            count += 1
            continue
        if count == 20001:
            break

        label = int(line.split(",")[0])
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

        data_tuple = (temp, label)
        data.append(data_tuple)
        count += 1

    return data


class DownstreamDataset(Dataset):

    def __init__(self, data, vocab, embeddings):
        self.data = data
        self.vocab = vocab
        self.embeddings = embeddings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx][0]
        label = self.data[idx][1] - 1  # return the index of the label
        sentence = [
            self.vocab[word] if word in self.vocab else self.vocab["<unk>"]
            for word in sentence
        ]
        return torch.LongTensor(sentence), torch.tensor(label)


def collate_fn(batch):
    sentences, labels = zip(*batch)
    sentences_len = [len(sentence) for sentence in sentences]
    sentences_pad = pad_sequence(sentences, batch_first=True, padding_value=float(1))
    labels = torch.Tensor(labels)
    return sentences_pad, labels, sentences_len


class DownstreamClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_labels, embeddings):
        super(DownstreamClassifier, self).__init__()

        self.embeddings = nn.Embedding.from_pretrained(
            torch.FloatTensor(embeddings), padding_idx=1, freeze=True
        )
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, num_labels)

    def forward(self, sentences):
        embeds = self.embeddings(sentences)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out[:, -1, :]
        return self.linear(lstm_out)


def train_model(
    model, train_loader, test_loader, num_epochs, optimizer, loss_fn, device
):
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch, (sentences, labels, sentences_len) in enumerate(train_loader):
            sentences, labels = sentences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sentences)
            loss = loss_fn(outputs, labels.view(-1).long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Training: Epoch {epoch}, Loss: {total_loss}, ", end = '')

        model.eval()
        total = 0
        correct = 0
        for sentences, labels, sentences_len in train_loader:
            sentences, labels = sentences.to(device), labels.to(device)
            outputs = model(sentences)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy: {correct/total:.4f}")


# def eval_model(model, test_loader, loss_fn, device):
#     model.to(device)
#     model.eval()
#     total = 0
#     correct = 0
#     test_loss = 0
#     with torch.no_grad():
#         for sentences, labels, sentences_len in test_loader:
#             sentences, labels = sentences.to(device), labels.to(device)
#             outputs = model(sentences)
#             loss = loss_fn(outputs, labels.view(-1).long())
#             test_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     print(f"Testing: Accuracy: {correct/total:.4f}")

from sklearn.metrics import classification_report


def eval_model(model, test_loader, loss_fn, device):
    model.to(device)
    model.eval()
    total = 0
    correct = 0
    test_loss = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for sentences, labels, sentences_len in test_loader:
            sentences, labels = sentences.to(device), labels.to(device)
            outputs = model(sentences)
            loss = loss_fn(outputs, labels.view(-1).long())
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.view(-1).cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    print(f"Testing: Accuracy: {correct/total:.4f}")
    print()
    print("Classification Report:")
    print(classification_report(all_labels, all_predictions))

    # make a confusion matrix

    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("skipgram_confusion_matrix_w8.png")



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = get_vocab()
    embeddings = torch.load("skip-gram-word-vectors.pt")
    embeddings = embeddings.numpy()

    train_data = get_data("ANLP-2/train.csv")
    test_data = get_data("ANLP-2/test.csv")

    train_dataset = DownstreamDataset(train_data, vocab, embeddings)
    test_dataset = DownstreamDataset(test_data, vocab, embeddings)

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn
    )

    model = DownstreamClassifier(300, 100, 4, embeddings)

    train_model(
        model,
        train_loader,
        test_loader,
        40,
        torch.optim.Adam(model.parameters(), lr=0.005),
        nn.CrossEntropyLoss(),
        device,
    )

    print()
    print()

    eval_model(model, test_loader, nn.CrossEntropyLoss(), device)

    # save model weights
    torch.save(model.state_dict(), "skip-gram-classification-model.pt")
