import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from models.modules.residual_block import Bottleneck, BasicBlock
from models.modules.pyramid_network import FeaturesPyramidNetwork
from models.functions.anchors import Anchors
from models.functions.focal_loss import FocalLoss
from models.modules.convlstm_fusion import ConvLSTM
from models.modules.non_local_aggregation import NonLocalAggregationModule
from models.modules.eventpillars import PillarFeatureNet, EventPillarsScatter


class DMANet(nn.Module):

    def __init__(self, in_channels, num_classes, block, layers):
        super(DMANet, self).__init__()
        self.inplanes = 64

        self.voxel_feature_extractor = PillarFeatureNet(num_input_features=3, use_norm=True,
                                                        num_filters=[in_channels], with_distance=False)

        self.middle_feature_extractor = EventPillarsScatter(output_shape=[1, 1, 512, 512, in_channels],
                                                            num_input_features=in_channels)

        self.input_layer = nn.Sequential(nn.Conv2d(in_channels*2, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2_lstm = ConvLSTM(input_size=128, hidden_size=128, kernel_size=3)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3_lstm = ConvLSTM(input_size=256, hidden_size=256, kernel_size=3)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.layer4_lstm = ConvLSTM(input_size=256, hidden_size=256, kernel_size=3)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.temporal_layer2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1),
                                             nn.BatchNorm2d(256))
        self.temporal_layer3 = nn.Sequential(nn.BatchNorm2d(256))
        self.temporal_layer4 = nn.Sequential(nn.BatchNorm2d(256))
        self.temporal_layer5 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
                                             nn.BatchNorm2d(256))
        self.temporal_layer6 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
                                             nn.BatchNorm2d(256))
        self.aggregation_layer = nn.Sequential(NonLocalAggregationModule(in_channels=128, reduction=2),
                                               NonLocalAggregationModule(in_channels=256, reduction=2),
                                               NonLocalAggregationModule(in_channels=256, reduction=2))
        self.fpn = FeaturesPyramidNetwork(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])  # FPN
        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        self.anchors = Anchors()
        self.focalLoss = FocalLoss()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        """Freeze BatchNorm layers."""
        print("\033[0;33m Starting to Freeze BatchNorm Layer! \033[0m")
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs, prev_states, prev_features):
        states = list()
        agg_features = list()

        pos_x_list, neg_x_list = inputs[0], inputs[1]
        pos_spatial_feature_list, neg_spatial_feature_list = [], []

        for pos_x, neg_x in zip(pos_x_list, neg_x_list):  # each batch
            pos_pillar_x, pos_pillar_y, pos_pillar_t, pos_num_points, pos_mask, pos_coors = pos_x
            neg_pillar_x, neg_pillar_y, neg_pillar_t, neg_num_points, neg_mask, neg_coors = neg_x
            pos_voxel_features = self.voxel_feature_extractor(pos_pillar_x, pos_pillar_y, pos_pillar_t, pos_num_points,
                                                              pos_mask)
            neg_voxel_features = self.voxel_feature_extractor(neg_pillar_x, neg_pillar_y, neg_pillar_t, neg_num_points,
                                                              neg_mask)
            pos_voxel_features = pos_voxel_features.squeeze(dim=0).squeeze(dim=-1).permute(1, 0)
            neg_voxel_features = neg_voxel_features.squeeze(dim=0).squeeze(dim=-1).permute(1, 0)

            # spatial_feature input size
            pos_spatial_features = self.middle_feature_extractor(pos_voxel_features, pos_coors.type(dtype=torch.int32))
            neg_spatial_features = self.middle_feature_extractor(neg_voxel_features, neg_coors.type(dtype=torch.int32))
            pos_spatial_feature_list.append(pos_spatial_features), neg_spatial_feature_list.append(neg_spatial_features)

        pos_spatial_feature = torch.cat([pos for pos in pos_spatial_feature_list], dim=0)
        neg_spatial_feature = torch.cat([neg for neg in neg_spatial_feature_list], dim=0)
        spatial_feature = torch.cat([pos_spatial_feature, neg_spatial_feature], dim=1)

        x = self.input_layer(spatial_feature)
        x1 = self.layer1(x)  # 128

        if prev_states is None:
            prev_states = [None] * 3

        x2 = self.layer2(x1)
        x2_lstm = self.layer2_lstm(x2, prev_states[0])
        states.append(x2_lstm)

        x3 = self.layer3(x2)
        x3_lstm = self.layer3_lstm(x3, prev_states[1])
        states.append(x3_lstm)

        x4 = self.layer4(x3)
        x4_lstm = self.layer4_lstm(x4, prev_states[2])
        states.append(x4_lstm)

        # short memory path
        fpn_features = [x2, x3, x4]

        if prev_features is None:
            agg_features = fpn_features
        else:
            for idx, (curr, prev) in enumerate(zip(fpn_features, prev_features)):
                batch_list = []
                for idy in range(curr.shape[0]):  # batch size
                    curr_f = curr[idy].unsqueeze(0)
                    prev_f = prev[idy].unsqueeze(0)
                    agg_feat = self.aggregation_layer[idx](curr_f, prev_f)
                    batch_list.append(agg_feat)
                batch_agg_feat = torch.cat([b for b in batch_list], dim=0)
                agg_features.append(batch_agg_feat)

        # [64, 32, 16, 8, 4]
        short_features = self.fpn(agg_features)

        # long memory path
        long_features = self.fpn([x2_lstm[0], x3_lstm[0], x4_lstm[0]])

        temporal_feature2 = self.temporal_layer2(x2_lstm[0])
        temporal_feature3 = self.temporal_layer3(x3_lstm[0])
        temporal_feature4 = self.temporal_layer4(x4_lstm[0])
        temporal_feature5 = self.temporal_layer5(temporal_feature4)
        temporal_feature6 = self.temporal_layer6(temporal_feature5)

        temporal_feature = [temporal_feature2, temporal_feature3, temporal_feature4, temporal_feature5, temporal_feature6]

        regression = torch.cat([self.regressionModel(s+l+t) for s, l, t in zip(short_features, long_features, temporal_feature)], dim=1)
        classification = torch.cat([self.classificationModel(s+l+t) for s, l, t in zip(short_features, long_features, temporal_feature)], dim=1)
        anchors = self.anchors(spatial_feature)
        return classification, regression, anchors, states, fpn_features, spatial_feature


class RegressionModel(nn.Module):
    # original num_anchors=3*3
    def __init__(self, num_features_in, num_anchors=5*3, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    # original num_anchors=3*3
    def __init__(self, num_features_in, num_anchors=3*5, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


def DMANet18(in_channels, num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DMANet(in_channels, num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def DMANet34(in_channels, num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DMANet(in_channels, num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def DMANet50(in_channels, num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DMANet(in_channels, num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def DMANet101(in_channels, num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DMANet(in_channels, num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def DMANet152(in_channels, num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DMANet(in_channels, num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model

# pretrained model
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
