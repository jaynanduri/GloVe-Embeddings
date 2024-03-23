import torch.optim as optim
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from model import GloveModel
from utils import load_20ng, doc2ind, load_glove

ng_text, ng_class = load_20ng()
embedding_matrix, word_to_idx = load_glove(ng_text)
docs_indices = [doc2ind(doc, word_to_idx) for doc in ng_text]

max_len = max(len(doc) for doc in docs_indices)

padded_docs = np.array([np.pad(doc, (0, max_len - len(doc)), 'constant', constant_values=0) for doc in docs_indices])

# Convert to torch tensors
padded_docs_tensor = torch.tensor(padded_docs, dtype=torch.long)
labels_tensor = torch.tensor(ng_class, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(padded_docs_tensor, labels_tensor, test_size=0.2, random_state=42)

# Create Dataloaders
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

batch_size = 64
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

# Model, Loss, and Optimizer
num_classes = np.unique(ng_class).shape[0]
model = GloveModel(embedding_matrix, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training Loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")

# Testing Loop
model.eval()
total_correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        total_correct += (predicted == y_batch).sum().item()
print(f'Accuracy: {100 * total_correct / total}%')

# saving updated embeddings to file
model.eval()  # Set the model to evaluation mode.
embeddings_list = []
labels_list = []

with torch.no_grad():  # No need to track gradients
    for texts, labels in test_loader:
        # Assuming texts are already in the correct tensor form (indices)
        embeddings = model.embedding(texts)  # Get the embeddings
        averaged_embeddings = embeddings.mean(dim=1)  # Average across the sequence
        embeddings_list.append(averaged_embeddings.cpu().numpy())
        labels_list.append(labels.cpu().numpy())

# Concatenate all batches
all_embeddings = np.concatenate(embeddings_list, axis=0)
all_labels = np.concatenate(labels_list, axis=0)

# Proceed with t-SNE and visualization
tsne = TSNE(n_components=2, random_state=0)
embeddings = tsne.fit_transform(all_embeddings)


def plot_embeddings(embeddings_2d: torch.Tensor, y: np.ndarray) -> None:
    plt.figure(figsize=(10, 8))
    # If y is not numeric, convert it to numeric values using LabelEncoder
    if np.issubdtype(y.dtype, np.number):
        classes = y
    else:
        le = LabelEncoder()
        classes = le.fit_transform(y)
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=classes, cmap='jet', alpha=0.7)
    plt.colorbar(scatter)
    plt.xlabel('TSNE axis 1')
    plt.ylabel('TSNE axis 2')
    plt.title('2D projection of embeddings')
    plt.show()


plot_embeddings(embeddings, all_labels)
