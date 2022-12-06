import torch
import torch.nn as nn
from models.functions.box_utils import BBoxTransform, ClipBoxes
from torchvision.ops import nms


class DMANet_Detector(nn.Module):
    def __init__(self, conf_threshold, iou_threshold):
        super(DMANet_Detector, self).__init__()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()

    def forward(self, classification, regression, anchors, img_batch):

        transformed_anchors = self.regressBoxes(anchors, regression)
        transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

        finalResult = [[], [], []]

        finalScores = torch.Tensor([])
        finalAnchorBoxesIndexes = torch.Tensor([]).long()
        finalAnchorBoxesCoordinates = torch.Tensor([])

        if torch.cuda.is_available():
            finalScores = finalScores.cuda()
            finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
            finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

        for i in range(classification.shape[2]):
            scores = torch.squeeze(classification[:, :, i])
            scores_over_thresh = (scores > self.conf_threshold)
            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just continue
                continue

            scores = scores[scores_over_thresh]
            anchorBoxes = torch.squeeze(transformed_anchors)
            anchorBoxes = anchorBoxes[scores_over_thresh]
            anchors_nms_idx = nms(anchorBoxes, scores, self.iou_threshold)

            finalResult[0].extend(scores[anchors_nms_idx])
            finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
            finalResult[2].extend(anchorBoxes[anchors_nms_idx])

            finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
            finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
            if torch.cuda.is_available():
                finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

            finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
            finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

        if len(finalScores):
            finalScores = finalScores.unsqueeze(-1)
            finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.type(torch.float32).unsqueeze(-1)
            finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates

            return torch.cat([finalAnchorBoxesCoordinates, finalScores, finalAnchorBoxesIndexes], dim=1)
        else:  # empty
            return torch.tensor([]).reshape(-1, 6)
