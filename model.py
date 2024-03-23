from torch import Tensor
import torch.nn as nn


class GloveModel(nn.Module):
    """
    This is a nn used to classify documents using pretrained GloVe Embeddings.
    """
    def __init__(self, embedding_matrix: Tensor, num_classes: int, *args, **kwargs) -> None:
        """
        Initialising Class params.
        :param embedding_matrix: Pre-trained Embedding matrix
        :param num_classes: num of classes in output
        """
        super(GloveModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.fc1 = nn.Linear(embedding_matrix.size(1), 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = x.mean(dim=1)  # Average across the sequence dimension
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



