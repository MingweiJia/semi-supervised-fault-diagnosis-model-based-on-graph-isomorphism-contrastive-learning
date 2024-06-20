import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

# 分数向量实现
class ScoreVector(nn.Module):
    def __init__(self, in_features, num_heads):
        super(ScoreVector, self).__init__()
        self.num_heads = num_heads
        self.query = nn.Linear(in_features, in_features * num_heads)
        self.key = nn.Linear(in_features, in_features * num_heads)
        self.value = nn.Linear(in_features, in_features * num_heads)

    def forward(self, x):
        batch_size, num_nodes, in_features = x.size()
        q = self.query(x).view(batch_size, num_nodes, self.num_heads, in_features)
        k = self.key(x).view(batch_size, num_nodes, self.num_heads, in_features)
        v = self.value(x).view(batch_size, num_nodes, self.num_heads, in_features)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (in_features ** 0.5)
        attention_matrix = F.softmax(attention_scores, dim=-1)
        return attention_matrix


# GCN实现
class GIN(nn.Module):
    def __init__(self, in_features, num_heads):
        super(GIN, self).__init__()
        self.num_heads = num_heads
        self.score_vector = ScoreVector(in_features, num_heads)
        self.linear = nn.Linear(num_heads, in_features)
        self.fc = nn.Linear(in_features, in_features)

    def normalize_adj(self, adj):
        adj = adj + torch.eye(adj.size(0)).to(adj.device)
        degree = torch.sum(adj, dim=1)
        degree_inv_sqrt = torch.diag(torch.pow(degree, -0.5))
        return torch.matmul(torch.matmul(degree_inv_sqrt, adj), degree_inv_sqrt)

    def forward(self, x, adj):
        batch_size, num_nodes, in_features = x.size()

        attention_matrix = self.score_vector(x)
        attention_matrix = attention_matrix.permute(0, 2, 3, 1)
        ScoreVector2 = self.linear(attention_matrix)
        noradj = self.normalize_adj(adj)

        ScoreVector_adj = noradj.unsqueeze(0).unsqueeze(
            -1) * ScoreVector2

        x_expanded = x.unsqueeze(1).repeat(1, num_nodes, 1, 1)

        ScoreVector_x = torch.sum(ScoreVector_adj * x_expanded, dim=1)

        out = torch.matmul(noradj, ScoreVector_x)
        out = self.fc(out)
        return F.relu(out)


class GIM(nn.Module):
    def __init__(self, in_features, hidden_features, num_heads, num_layers):
        super(GIM, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        self.layers.append(GIN(in_features, num_heads))

        for _ in range(num_layers - 2):
            self.layers.append(GIN(hidden_features, num_heads))

        self.layers.append(GIN(hidden_features, num_heads))

        self.linear = nn.Linear(in_features, hidden_features)

    def forward(self, x, adj):
        x = self.linear(x)
        for layer in self.layers:
            x = layer(x, adj)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ImportanceCalculator(nn.Module):
    def __init__(self, in_features):
        super(ImportanceCalculator, self).__init__()
        self.fc = nn.Linear(in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        importance = self.fc(x)
        importance = self.sigmoid(importance)
        return importance.squeeze(-1)


# MoCo对比学习实现
class GICL(nn.Module):
    def __init__(self, in_features, hidden_features, num_heads, num_layers, num_classes, dim=128, K=65536, m=0.999, T=0.07,
                 noise_std=0.01):
        super(GICL, self).__init__()
        self.encoder_q = GIM(in_features, in_features, num_heads, num_layers)
        self.encoder_k = copy.deepcopy(self.encoder_q)
        self.classifier = ClassificationHead(in_features, hidden_features, num_classes)
        self.importance_calculator = ImportanceCalculator(in_features)
        self.dim = dim
        self.K = K
        self.m = m
        self.T = T
        self.noise_std = noise_std

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update_encoder_k(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            # 为encoder_k参数添加高斯噪声
            param_k.data += torch.randn_like(param_k.data) * self.noise_std

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k, adj_q, adj_k):
        importance = self.importance_calculator(im_q)  # 重要度形状：(batch_size, num_nodes)
        importance = importance + 1  # 加1
        im_q = im_q * importance.unsqueeze(-1)  # 重要度与输入相乘

        q = self.encoder_q(im_q, adj_q)
        q = F.normalize(q.mean(dim=1), dim=1)  # 对每个节点特征取平均

        with torch.no_grad():
            self.momentum_update_encoder_k()
            k = self.encoder_k(im_k, adj_k)
            k = F.normalize(k.mean(dim=1), dim=1)  # 对每个节点特征取平均

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        self.dequeue_and_enqueue(k)

        class_logits = self.classifier(q)

        return logits, labels, class_logits

if __name__ == '__main__':

    in_features = 50
    hidden_features = 64
    num_heads = 8
    num_layers = 3
    num_classes = 7

    moco = GICL(in_features, hidden_features, num_heads, num_layers, num_classes, dim=in_features, noise_std=0.01)
    total_params = sum(p.numel() for p in moco.parameters())
    print(f'Total number of parameters: {total_params}')

    batch_size = 32
    num_nodes = 24
    x_q = torch.randn(batch_size, num_nodes, in_features)
    x_k = torch.randn(batch_size, num_nodes, in_features)
    adj_q = torch.eye(num_nodes)
    adj_k = torch.eye(num_nodes)
    class_labels = torch.randint(0, num_classes, (batch_size,))

    optimizer = optim.SGD(moco.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)

    for epoch in range(10):
        moco.train()
        optimizer.zero_grad()

        logits, labels, class_logits = moco(x_q, x_k, adj_q, adj_k)
        contrastive_loss = F.cross_entropy(logits, labels)

        if torch.rand(1).item() < 0.2:
            classification_loss = F.cross_entropy(class_logits, class_labels)
            loss = contrastive_loss + classification_loss
        else:
            loss = contrastive_loss

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
