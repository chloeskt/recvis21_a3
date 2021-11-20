import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HardBatchMiningTripletLoss(torch.nn.Module):
    def __init__(self, margin=0.3):
        super(HardBatchMiningTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = torch.nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        input1 = inputs
        input2 = inputs.transpose(0, 1)
        matrix_product = torch.matmul(input1, input2)
        diag = torch.diag(matrix_product)
        distance_matrix = diag.unsqueeze(0) - 2.0 * matrix_product + diag.unsqueeze(1)
        distance_matrix = distance_matrix.clamp(min=1e-12).sqrt()

        distance_positive_pairs, distance_negative_pairs = [], []
        for i in range(n):
            current_label = targets[i].item()
            mask = targets.eq(current_label)
            distance_positive = torch.max(
                torch.masked_select(distance_matrix[i, :], mask)
            )
            distance_negative = torch.min(
                torch.masked_select(distance_matrix[i, :], torch.logical_not(mask))
            )
            distance_positive_pairs.append(distance_positive)
            distance_negative_pairs.append(distance_negative)

        distance_positive_pairs = torch.tensor(distance_positive_pairs, device=device)
        distance_negative_pairs = torch.tensor(distance_negative_pairs, device=device)

        y = torch.ones_like(distance_negative_pairs)
        return self.ranking_loss(distance_negative_pairs, distance_positive_pairs, y)


class CombinedLoss(object):
    def __init__(self, weight_triplet=1.0, weight_ce=1.5):
        super(CombinedLoss, self).__init__()
        self.triplet_loss = HardBatchMiningTripletLoss()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.weight_triplet = weight_triplet
        self.weight_ce = weight_ce

    def __call__(self, logits, features, gt_pids):
        loss = 0.0
        loss_summary = {}
        if self.weight_triplet > 0.0:
            loss_t = self.triplet_loss(features, gt_pids) * self.weight_triplet
            loss += loss_t
            loss_summary["Triplet Loss"] = loss_t

        if self.weight_ce > 0.0:
            loss_ce = self.cross_entropy(logits, gt_pids) * self.weight_ce
            loss += loss_ce
            loss_summary["CE Loss"] = loss_ce

        loss_summary["Loss"] = loss
        return loss, loss_summary


if __name__ == "__main__":
    criterion = CombinedLoss(0.3, 0.5, 2.0)
