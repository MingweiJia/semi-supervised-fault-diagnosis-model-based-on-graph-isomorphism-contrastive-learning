import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
from model import GICL


class TEDataset(Dataset):
    def __init__(self, data_dir, split):
        self.data_dir = data_dir
        self.split = split
        self.data = self.load_data()

    def load_data(self):
        # Load data from CSV files or other formats
        data = []
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('.csv') and self.split in file_name:
                file_path = os.path.join(self.data_dir, file_name)
                df = pd.read_csv(file_path)
                data.append(df.values)
        return np.concatenate(data, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        x = sample[:-1]  # All columns except the last one as features
        y = sample[-1]  # The last column as label
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_q, y_q in test_loader:
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
    test_dataset = TEDataset(data_dir, split='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    in_features = 50
    hidden_features = 64
    num_heads = 8
    num_layers = 3
    num_classes = 7

    model = GICL(in_features, hidden_features, num_heads, num_layers, num_classes, dim=in_features, noise_std=0.01)
    model.load_state_dict(torch.load('path_to_saved_model'))

    test_accuracy = test(model, test_loader)
    print(f'Test Accuracy: {test_accuracy:.2f}%')


