"""
Example usage: CUDA_VISIBLE_DEVICES=1, python train_DMANet.py --settings_file "config/settings.yaml"
"""
import argparse
import os
import abc
import tqdm
import torch
import math
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

import dataloader.dataset
from models.functions.focal_loss import FocalLoss
from models.functions.box_utils import box_iou
from models.modules import dmanet_network
from models.modules.dmanet_detector import DMANet_Detector
from dataloader.loader import Loader
from models.functions.smooth_l1_loss import Smooth_L1_Loss
from config.settings import Settings
from utils.metrics import ap_per_class
from apex import amp
from models.functions.warmup import WarmUpLR
torch.multiprocessing.set_sharing_strategy('file_system')


class AbstractTrainer(abc.ABC):
    def __init__(self, settings):
        self.settings = settings
        self.model = None
        self.scheduler = None
        self.nr_classes = None   # numbers of classes
        self.train_loader = None
        self.val_loader = None
        self.nr_train_epochs = None
        self.nr_val_epochs = None
        self.train_file_indexes = None
        self.val_file_indexes = None
        self.object_classes = None
        self.train_sampler = None

        self.dataset_builder = dataloader.dataset.getDataloader(self.settings.dataset_name)  # Prophesee
        self.dataset_loader = Loader
        self.writer = SummaryWriter(self.settings.ckpt_dir)

        self.createDatasets()  # train_dataset and val_dataset

        self.buildModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.settings.init_lr)
        # mixed precision training
        amp.register_float_function(torch, "sigmoid")
        amp.register_float_function(torch, "softmax")
        amp.register_float_function(torch, "matmul")
        amp.register_float_function(torch.nn, "MaxPool2d")
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")

        self.warmup_schedular = WarmUpLR(self.optimizer, len(self.train_loader)*self.settings.warm)
        self.train_schedular = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                          T_max=len(self.train_loader)*(self.settings.epoch-self.settings.warm),
                                                                          eta_min=self.settings.init_lr*0.1)

        self.batch_step = 0
        self.epoch_step = 0
        self.training_loss = 0
        self.training_accuracy = 0
        self.max_validation_mAP = 0
        self.softmax = nn.Softmax(dim=-1)
        self.smooth_l1_loss = Smooth_L1_Loss(beta=0.11, reduction="sum")

        # tqdm progress bar
        self.pbar = None

        if settings.resume_training:
            self.loadCheckpoint(self.settings.resume_ckpt_file)

    @abc.abstractmethod
    def buildModel(self):
        """Model is constructed in child class"""
        pass

    def createDatasets(self):
        """
        Creates the validation and the training data based on the lists specified in the config/settings.yaml file.
        """
        train_dataset = self.dataset_builder(self.settings.dataset_path,
                                             self.settings.object_classes,
                                             self.settings.height,
                                             self.settings.width,
                                             mode="training",
                                             voxel_size=self.settings.voxel_size,
                                             max_num_points=self.settings.max_num_points,
                                             max_voxels=self.settings.max_voxels,
                                             resize=self.settings.resize,
                                             num_bins=self.settings.num_bins)
        self.train_file_indexes = train_dataset.file_index()
        self.nr_train_epochs = train_dataset.nr_samples // self.settings.batch_size + 1
        self.nr_classes = train_dataset.nr_classes
        self.object_classes = train_dataset.object_classes

        val_dataset = self.dataset_builder(self.settings.dataset_path,
                                           self.settings.object_classes,
                                           self.settings.height,
                                           self.settings.width,
                                           mode="validation",
                                           voxel_size=self.settings.voxel_size,
                                           max_num_points=self.settings.max_num_points,
                                           max_voxels=self.settings.max_voxels,
                                           resize=self.settings.resize,
                                           num_bins=self.settings.num_bins)
        self.val_file_indexes = val_dataset.file_index()
        self.nr_val_epochs = val_dataset.nr_samples

        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(list(range(len(train_dataset))))

        self.train_loader = self.dataset_loader(train_dataset, mode="training",
                                                batch_size=self.settings.batch_size,
                                                num_workers=self.settings.num_cpu_workers, pin_memory=False,
                                                drop_last=True, sampler=self.train_sampler,
                                                data_index=self.train_file_indexes)

        self.val_loader = self.dataset_loader(val_dataset, mode="validation",
                                              batch_size=self.settings.batch_size // self.settings.batch_size,
                                              num_workers=self.settings.num_cpu_workers, pin_memory=False,
                                              drop_last=True, sampler=None,
                                              data_index=self.val_file_indexes)

    def storeLossesObjectDetection(self, loss_list):
        """Writes the different losses to tensorboard"""
        loss_names = ["Confidence_Loss", "Location_Loss", "Overall_Loss"]

        for idx in range(len(loss_list)):
            loss_value = loss_list[idx].data.cpu().numpy()
            self.writer.add_scalar("TrainingLoss/" + loss_names[idx], loss_value, self.batch_step)

    def storeClassmAP(self, map_list):
        class_names = self.settings.object_classes
        for idx in range(len(class_names)):
            class_map = map_list[idx]
            class_name = class_names[idx] + "_mAP"
            self.writer.add_scalar("Validation/"+class_name, class_map, self.epoch_step)
        self.writer.add_scalar("Validation/mAP0.5", map_list.mean(), self.epoch_step)

    def getLearningRate(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def loadCheckpoint(self, filename):
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            self.epoch_step = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

            print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint["epoch"]))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

    def saveCheckpoint(self):
        file_path = os.path.join(self.settings.ckpt_dir, "model_step_" + str(self.epoch_step) + ".pth")
        torch.save({"state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "epoch": self.epoch_step}, file_path)


class DMANetDetection(AbstractTrainer):
    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        """
        Create boolean mask by actually number of a padded tensor.
        :param actual_num:
        :param max_num:
        :param axis:
        :return: [type]: [description]
        """
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        # paddings_indicator shape : [batch_size, max_num]
        return paddings_indicator

    def process_pillar_input(self, events, idx, idy):
        pillar_x = events[idy][idx][0][..., 0].unsqueeze(0).unsqueeze(0)
        pillar_y = events[idy][idx][0][..., 1].unsqueeze(0).unsqueeze(0)
        pillar_t = events[idy][idx][0][..., 2].unsqueeze(0).unsqueeze(0)
        coors = events[idy][idx][1]
        num_points_per_pillar = events[idy][idx][2].unsqueeze(0)
        num_points_per_a_pillar = pillar_x.size()[3]
        mask = self.get_paddings_indicator(num_points_per_pillar, num_points_per_a_pillar, axis=0)
        mask = mask.permute(0, 2, 1).unsqueeze(1).type_as(pillar_x)
        input = [pillar_x.cuda(), pillar_y.cuda(), pillar_t.cuda(),
                     num_points_per_pillar.cuda(), mask.cuda(), coors.cuda()]
        return input

    def buildModel(self):
        """Creates the specified model"""
        if self.settings.depth == 18:
            self.model = dmanet_network.DMANet18(in_channels=self.settings.nr_input_channels,
                                                       num_classes=len(self.settings.object_classes), pretrained=False)
        elif self.settings.depth == 34:
            self.model = dmanet_network.DMANet34(in_channels=self.settings.nr_input_channels,
                                                       num_classes=len(self.settings.object_classes), pretrained=False)
        elif self.settings.depth == 50:
            self.model = dmanet_network.DMANet50(in_channels=self.settings.nr_input_channels,
                                                       num_classes=len(self.settings.object_classes), pretrained=False)
        elif self.settings.depth == 101:
            self.model = dmanet_network.DMANet101(in_channels=self.settings.nr_input_channels,
                                                        num_classes=len(self.settings.object_classes), pretrained=False)
        elif self.settings.depth == 152:
            self.model = dmanet_network.DMANet152(in_channels=self.settings.nr_input_channels,
                                                        num_classes=len(self.settings.object_classes), pretrained=False)
        else:
            raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

        self.params_initialize()
        self.model.to(self.settings.gpu_device)

        if self.settings.use_pretrained:
            self.loadPretrainedWeights()

    def params_initialize(self):
        print("\033[0;33m Starting to initialize parameters! \033[0m")
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01
        self.model.classificationModel.output.weight.data.fill_(0)
        self.model.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
        self.model.regressionModel.output.weight.data.fill_(0)
        self.model.regressionModel.output.bias.data.fill_(0)

    def loadPretrainedWeights(self):
        """Loads pretrained model weights"""
        print("\033[0;33m Using pretrained model! \033[0m")
        checkpoint = torch.load(self.settings.pretrained_model)
        try:
            pretrained_dict = checkpoint["state_dict"]
        except KeyError:
            pretrained_dict = checkpoint["model"]

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if "input_layer." not in k}
        self.model.load_state_dict(pretrained_dict, strict=False)

    def train(self):
        """Main training and validation loop"""

        while self.epoch_step <= self.settings.epoch:
            self.trainEpoch()
            self.validationEpoch()

            self.epoch_step += 1

    def trainEpoch(self):
        self.pbar = tqdm.tqdm(total=self.nr_train_epochs, unit="Batch", unit_scale=True,
                              desc="Epoch: {}".format(self.epoch_step))
        self.model.train()
        focal_loss = FocalLoss()

        for i_batch, sample_batched in enumerate(self.train_loader):
            if self.epoch_step < self.settings.warm:
                self.warmup_schedular.step()
            bounding_box, pos_events, neg_events = sample_batched
            prev_states, prev_features = None, None
            batch_reg_loss, batch_cls_loss, batch_total_loss = 0, 0, 0
            self.optimizer.zero_grad()  # reset gradient

            for idx in range(self.settings.seq_len):
                pos_input_list, neg_input_list = [], []
                bounding_box_now, bounding_box_next = [], []
                for idy in range(self.settings.batch_size):
                    # process positive/negative events
                    pos_input = self.process_pillar_input(pos_events, idx, idy)
                    neg_input = self.process_pillar_input(neg_events, idx, idy)

                    pos_input_list.append(pos_input), neg_input_list.append(neg_input)

                    # labels
                    mask_now = (bounding_box[:, -1] == (idy * self.settings.seq_len + idx))
                    mask_next = (bounding_box[:, -1] == (idy * self.settings.seq_len + idx + 1))
                    bbox_now = bounding_box[mask_now][:, :-1]
                    bbox_next = bounding_box[mask_next][:, :-1]
                    bounding_box_now.append(bbox_now), bounding_box_next.append(bbox_next)

                classification, regression, anchors, prev_states, prev_features, _ = \
                    self.model([pos_input_list, neg_input_list], prev_states=prev_states, prev_features=prev_features)
                cls_loss, reg_loss = focal_loss(classification, regression, anchors, torch.tensor(bounding_box_now).cuda())
                cls_loss, reg_loss = cls_loss.mean(), reg_loss.mean()
                batch_cls_loss += cls_loss
                batch_reg_loss += reg_loss
                batch_total_loss += (cls_loss + reg_loss)

            if batch_total_loss != 0:
                with amp.scale_loss(batch_total_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()  # update parameters of network according to gradient
            del cls_loss, reg_loss
            # to visualize loss curve better
            cls_loss = torch.tensor(0, dtype=torch.float32) if batch_cls_loss == 0 else batch_cls_loss
            reg_loss = torch.tensor(0, dtype=torch.float32) if batch_reg_loss == 0 else batch_reg_loss
            total_loss = torch.tensor(0, dtype=torch.float32) if batch_total_loss == 0 else batch_total_loss
            self.pbar.set_postfix(Conf=cls_loss.item(), Loc=reg_loss.item(), Total_Loss=total_loss.item())
            loss_list = [cls_loss, reg_loss, total_loss]
            # Write losses statistics
            self.storeLossesObjectDetection(loss_list)
            self.pbar.update(1)
            self.batch_step += 1
            self.writer.add_scalar("Training/Learning_Rate_batch", self.getLearningRate(), self.batch_step)

            if self.epoch_step >= self.settings.warm:
                self.train_schedular.step((self.epoch_step - self.settings.warm) * len(self.train_loader) + i_batch)
        self.writer.add_scalar("Training/Learning_Rate", self.getLearningRate(), self.epoch_step)

        self.pbar.close()

    def validationEpoch(self):
        self.pbar = tqdm.tqdm(total=self.nr_val_epochs, unit="Batch", unit_scale=True)
        self.model.eval()
        dmanet_detector = DMANet_Detector(conf_threshold=0.1, iou_threshold=0.5)

        iouv = torch.linspace(0.5, 0.95, 10).to(self.settings.gpu_device)
        niou = iouv.numel()  # len(iouv)
        seen = 0
        precision, recall, f1_score, m_precision, m_recall, map50, map = 0., 0., 0., 0., 0., 0., 0.
        stats, ap, ap_class = [], [], []

        scale = torch.tensor([self.settings.resize, self.settings.resize, self.settings.resize, self.settings.resize],
                             dtype=torch.float32).to(self.settings.gpu_device)

        prev_states = None
        for i_batch, sample_batched in enumerate(self.val_loader):
            prev_features = None
            detection_result = []  # detection result for computing mAP
            bounding_box, pos_events, neg_events = sample_batched

            with torch.no_grad():
                for idx in range(self.settings.seq_len):
                    pos_input_list, neg_input_list = [], []
                    for idy in range(self.settings.batch_size // self.settings.batch_size):
                        # process positive/negative events
                        pos_input = self.process_pillar_input(pos_events, idx, idy)
                        neg_input = self.process_pillar_input(neg_events, idx, idy)

                        pos_input_list.append(pos_input), neg_input_list.append(neg_input)
                    classification, regression, anchors, prev_states, prev_features, pseudo_img = \
                        self.model([pos_input_list, neg_input_list], prev_states=prev_states, prev_features=prev_features)
                    # [coords, scores, labels]
                    out = dmanet_detector(classification, regression, anchors, pseudo_img)
                    detection_result.append(out)
            self.pbar.update(1)

            for si, pred in enumerate(detection_result):
                bbox = bounding_box[bounding_box[:, -1] == si]  # each batch
                np_labels = bbox[bbox[:, -2] != -1.]
                np_labels = np_labels[:, [4, 0, 1, 2, 3]]  # [cls, coords]
                labels = torch.from_numpy(np_labels).to(self.settings.gpu_device)

                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue
                # predictions
                predn = pred.clone()
                predn[:, :4] /= self.settings.resize  # percent coordinates
                predn[:, :4] *= scale  # absolute coordinates, 1280x720

                # assign all predictions as incorrect
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool).to(self.settings.gpu_device)
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]
                    # target boxes
                    tbox = labels[:, 1:5]
                    # per target class
                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices
                        # search for detections
                        if pi.shape[0]:
                            # prediction to target ious
                            ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices
                            # append detections
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]  # detected target
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    if len(detected) == nl:  # all targets already located in images
                                        break
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
        self.pbar.close()

        # Directories
        save_dir = os.path.join(self.settings.save_dir, "det_result")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)   # make dir
        names = {k: v for k, v in enumerate(self.object_classes, start=0)}  # {0: 'Pedestrian', ...}
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            precision, recall, ap, f1_score, ap_class = ap_per_class(*stats, plot=False, save_dir=save_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(axis=1)
            m_precision, m_recall, map50, map = precision.mean(), recall.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=self.nr_classes-1)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        pf = "%8s" + "%18i" * 2 + "%19.3g" * 4  # print format
        print("\033[0;31m    Class            Events              Labels           Precision           Recall          "
              "   mAP@0.5           mAP@0.5:0.95 \033[0m")
        print(pf % ("all", seen, nt.sum(), m_precision, m_recall, map50, map))

        # Print results per class
        if len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (names[c], seen, nt[c], precision[i], recall[i], ap50[i], ap[i]))
                self.writer.add_scalar("Validation/" + names[c], ap50[i], self.epoch_step)
        self.writer.add_scalar("Validation/mAP@0.5", map50, self.epoch_step)
        # if self.max_validation_mAP < map50:
        #     self.max_validation_mAP = map50
        self.saveCheckpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train network.")
    parser.add_argument("--settings_file", type=str, default="/home/wds/DMANet/config/settings.yaml",
                        help="Path to settings yaml")

    args = parser.parse_args()
    settings_filepath = args.settings_file
    settings = Settings(settings_filepath, generate_log=True)

    if settings.model_name == "dmanet":
        trainer = DMANetDetection(settings)
    else:
        raise ValueError("Model name %s specified in the settings file is not implemented" % settings.model_name)

    trainer.train()

