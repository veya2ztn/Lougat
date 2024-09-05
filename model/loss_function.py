from torchmetrics.detection import IntersectionOverUnion

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
        self.metric = IntersectionOverUnion()

    def forward(self, pred, target):
        B1, D1 = pred.shape
        B2, D2 = target.shape
        assert B1 == B2
        assert D1 == D2
        assert D1 == 4
        bbox_labels = torch.arange(pred.size(0)).to(pred)
        preds  = [{"boxes" :   pred,"labels": bbox_labels,}]
        target = [{"boxes" : target,"labels": bbox_labels,}]
        return self.metric(preds, target)
