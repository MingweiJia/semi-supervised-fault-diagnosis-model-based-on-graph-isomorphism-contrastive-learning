import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os
import copy
from GICL import GICL


class TEDataset(Dataset):
    def __init__(self, data_dir, split):
        self.data_dir = data_dir
        self.split = split
        self.data = self.load_data()

    def load_data(self):
        data = []
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('.csv') and self.split in file_name:
                file_path = os.path.join(self.data_dir, file_name)
                df = pd.read_csv(file_path)
                data.append(df.values)
        return np.concatenate(data, axis=0)[:, :, 40:, [*range(3, 22), *range(23, 30), *range(44, 48), *range(49, 52), *range(53, 55)]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        x = sample[:-1]
        y = sample[-1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def train(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0.0
    for batch_idx, (x_q, y_q) in enumerate(train_loader):
        x_k = x_q.clone()
        adj_q = torch.tensor(np.array(pd.read_csv(r'...', header=None)))
        adj_k = adj_q.clone()

        optimizer.zero_grad()

        logits, labels, class_logits = model(x_q, x_k, adj_q, adj_k)
        contrastive_loss = F.cross_entropy(logits, labels)
        classification_loss = F.cross_entropy(class_logits, y_q)
        loss = contrastive_loss + classification_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    return total_loss / len(train_loader)


def validate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_q, y_q in val_loader:
            x_k = x_q.clone()
            adj_q = torch.tensor(np.array(pd.read_csv(r'...', header=None)))
            adj_k = adj_q.clone()

            _, _, class_logits = model(x_q, x_k, adj_q, adj_k)
            _, predicted = torch.max(class_logits, 1)
            total += y_q.size(0)
            correct += (predicted == y_q).sum().item()
    accuracy = 100 * correct / total
    return accuracy


if __name__ == '__main__':
    data_dir = '...'
    train_dataset = TEDataset(data_dir, split='train')
    val_dataset = TEDataset(data_dir, split='val')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    in_features = 35
    hidden_features = 64
    num_heads = 8
    num_layers = 4
    num_classes = 21

    model = GICL(in_features, hidden_features, num_heads, num_layers, num_classes, dim=in_features, noise_std=0.01)
    optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)

    num_epochs = 100

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, epoch)
        val_accuracy = validate(model, val_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
