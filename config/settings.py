import os
import time
import yaml
import torch
import shutil


class Settings:
    def __init__(self, settings_yaml, generate_log=True):
        assert os.path.isfile(settings_yaml), settings_yaml

        with open(settings_yaml, 'r') as stream:
            settings = yaml.load(stream, yaml.Loader)

            # --- hardware ---
            hardware = settings["hardware"]
            gpu_device = hardware["gpu_device"]

            self.gpu_device = torch.device("cpu") if gpu_device == "cpu" else torch.device("cuda:" + str(gpu_device))

            self.num_cpu_workers = hardware["num_cpu_workers"]
            if self.num_cpu_workers < 0:
                self.num_cpu_workers = os.cpu_count()

            # --- Model ---
            model = settings["model"]
            self.model_name = model["model_name"]

            # --- dataset ---
            dataset = settings["dataset"]
            self.dataset_name = dataset["name"]
            self.event_representation = dataset["event_representation"]
            if self.dataset_name == "Prophesee":
                dataset_specs = dataset["prophesee"]
            self.dataset_path = dataset_specs["dataset_path"]
            assert os.path.isdir(self.dataset_path)
            self.object_classes = dataset_specs["object_classes"]
            self.depth = dataset_specs["depth"]
            self.height = dataset_specs["height"]
            self.width = dataset_specs["width"]
            self.resize = dataset_specs["resize"]
            self.voxel_size = dataset_specs["voxel_size"]
            self.max_num_points = dataset_specs["max_num_points"]
            self.max_voxels = dataset_specs["max_voxels"]
            self.num_bins = dataset_specs["num_bins"]
            self.nr_input_channels = dataset_specs["nr_input_channels"]

            # --- checkpoint ---
            checkpoint = settings["checkpoint"]
            self.resume_training = checkpoint["resume_training"]
            assert isinstance(self.resume_training, bool)
            self.save_dir = checkpoint["save_dir"]
            self.resume_ckpt_file = checkpoint["resume_file"]
            self.use_pretrained = checkpoint["use_pretrained"]
            self.pretrained_model = checkpoint["pretrained_model"]

            # --- directories ---
            directories = settings["dir"]
            log_dir = directories["log"]

            # --- logs ---
            if generate_log:
                timestr = time.strftime("%Y%m%d-%H%M%S")
                log_dir = os.path.join(log_dir, timestr)
                os.makedirs(log_dir)
                settings_copy_filepath = os.path.join(log_dir, "settings.yaml")
                shutil.copyfile(settings_yaml, settings_copy_filepath)
                self.ckpt_dir = os.path.join(log_dir, "checkpoints")
                os.mkdir(self.ckpt_dir)
                self.vis_dir = os.path.join(log_dir, "visualization")
                os.mkdir(self.vis_dir)
            else:
                self.ckpt_dir = os.path.join(log_dir, "checkpoints")
                self.vis_dir = os.path.join(log_dir, "visualization")

            # --- optimization ---
            optimization = settings["optim"]
            self.seq_len = optimization["seq_len"]
            self.epoch = optimization["epoch"]
            self.batch_size = optimization["batch_size"]
            self.init_lr = float(optimization["init_lr"])
            self.exponential_decay = float(optimization["exponential_decay"])
            self.warm = optimization["warm"]
            self.tbptt = optimization["tbptt"]
