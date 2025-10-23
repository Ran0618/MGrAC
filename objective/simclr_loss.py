import torch
import torch.nn.functional as F

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        # 生成每个样本标签的数组两遍
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(2)], dim=0)

        # 将数组分别转换为行向量和列向量，并进行广播比较生成bool数组，True的位置记录了正样本，False记录了负样本
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        # 特征形状：(2*batch-size , mlp_dim),归一化后计算余弦相似性
        features = F.normalize(features, dim=1)

        # 计算每个样本之间的相似性矩阵
        similarity_matrix = torch.matmul(features, features.T)
        # 确保形状一致
        assert similarity_matrix.shape == (
            self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        assert similarity_matrix.shape == labels.shape

        # 丢弃标签和相似性矩阵的主对角线，因为这部分自己和自己的相似度，我们要找和另一个视角的相似度
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # 选取和正样本的相似度，变成一个维数组，再给他变回（2N，1）的的矩阵
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # 选取和负样本的相似度，最终形状（2N，2N-2）
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        # 类似于MoCo把正样本放在第一个，再利用一个全0标签数组做后续交叉熵损失
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature

        loss = self.criterion(logits, labels)

        return loss