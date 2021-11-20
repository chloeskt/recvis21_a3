import torch


class HardBatchMiningTripletLoss(torch.nn.Module):
    """Triplet loss with hard positive/negative mining of samples in a batch.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(HardBatchMiningTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = torch.nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
        """
        n = inputs.size(0)

        # TASK: Compute the pairwise euclidean distance between all n feature vectors.
        # Hint: We recommend computing the actual euclidean distance (not squared).
        # For numerical stability, you can do sth. like:
        # distance_matrix = distance_matrix.clamp(min=1e-12).sqrt()
        input1 = inputs
        input2 = inputs.transpose(0, 1)
        matrix_product = torch.matmul(input1, input2)
        diag = torch.diag(matrix_product)
        distance_matrix = diag.unsqueeze(0) - 2.0 * matrix_product + diag.unsqueeze(1)
        distance_matrix = distance_matrix.clamp(min=1e-12).sqrt()

        # TASK: For each sample (image), find the hardest positive and hardest negative sample.
        # The targets are a vector that encode the class label for each of the n samples.
        # Pairs of samples with the SAME class can form a positive sample.
        # Pairs of samples with a DIFFERENT class can form a negative sample.
        #
        # For this task, you will need to loop over all samples, and for each one
        # find the hardest positive sample and the hardest negative sample.
        # The distances are then added to the following lists.
        # Please think about what hardest means for positive and negative pairs.
        # Reminder: Positive pairs should be as close as possible, while
        # negative pairs should be quite far apart.

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

        # TASK: Convert the created lists into 1D pytorch tensors. Please never
        # convert the tensors to numpy or raw python format, as you want to backpropagate
        # the loss, i.e., the above lists should only contain pytorch tensors.
        # Hint: Checkout the pytorch documentation.
        distance_positive_pairs = torch.tensor(distance_positive_pairs, device=device)
        distance_negative_pairs = torch.tensor(distance_negative_pairs, device=device)

        # The ranking loss will compute the triplet loss with the margin.
        # loss = max(0, -1*(neg_dist - pos_dist) + margin)
        # This is done already, no need to change anything.
        y = torch.ones_like(distance_negative_pairs)
        return self.ranking_loss(distance_negative_pairs, distance_positive_pairs, y)


class CombinedLoss(object):
    def __init__(self, margin=0.3, weight_triplet=1.0, weight_ce=1.5):
        super(CombinedLoss, self).__init__()
        self.triplet_loss = HardBatchMiningTripletLoss()  # <--- Your code is used here!
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


criterion = CombinedLoss(0.3, 0.5, 2.0)
