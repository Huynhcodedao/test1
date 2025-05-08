import os
import torch
import numpy as np
from utils.data_augment import WiderFacePreprocess
from model.config import INPUT_SIZE
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class WiderFaceDataset(Dataset):
    """
    Wider Face custom dataset.
    Args:
        root_path (string): Path to dataset directory
        is_train (bool): Train dataset or test dataset
        transform (function): whether to apply the data augmentation scheme
                mentioned in the paper. Only applied on the train split.
    """

    def __init__(self, root_path, input_size=INPUT_SIZE, is_train=True, label_file='labels.txt'):
        self.ids = []
        self.label_dict = {}
        self.transform = WiderFacePreprocess(image_size=input_size)
        self.is_train = is_train

        # Đặt đường dẫn đến thư mục hình ảnh và nhãn
        self.image_path = os.path.join(root_path, 'image')
        self.label_path = os.path.join(root_path, label_file)

        # Đọc labels.txt
        print(f"Đang cố mở tệp: {self.label_path}")
        with open(self.label_path, 'r') as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            img_name = lines[i].strip()
            num_bbox = int(lines[i+1].strip())
            bboxes = []
            for j in range(num_bbox):
                bboxes.append(lines[i+2+j].strip().split())
            base = os.path.splitext(os.path.basename(img_name))[0]
            self.label_dict[base] = bboxes
            i = i + 2 + num_bbox

        # Lấy danh sách hình ảnh
        all_images = [f[:-4] for f in os.listdir(self.image_path) if f.endswith('.png')]
        
        # Chia train/val (80% train, 20% val)
        split_idx = int(0.8 * len(all_images))
        if is_train:
            self.ids = all_images[:split_idx]
        else:
            self.ids = all_images[split_idx:]

        print(f"Số lượng mẫu {'train' if is_train else 'val'}: {len(self.ids)}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_name = self.ids[index]
        img_path = os.path.join(self.image_path, f"{img_name}.png")
        img = Image.open(img_path)
        img = np.array(img)

        # Lấy nhãn từ label_dict
        bboxes = self.label_dict.get(img_name, [])
        annotations = np.zeros((len(bboxes), 15))

        if len(bboxes) == 0:
            return img, annotations

        for idx, bbox in enumerate(bboxes):
            bbox = [float(x) for x in bbox]
            # bbox
            annotations[idx, 0] = bbox[0]               # x1
            annotations[idx, 1] = bbox[1]               # y1
            annotations[idx, 2] = bbox[0] + bbox[2]     # x2
            annotations[idx, 3] = bbox[1] + bbox[3]     # y2

            if self.is_train:
                # landmarks
                annotations[idx, 4] = bbox[4]           # l0_x
                annotations[idx, 5] = bbox[5]           # l0_y
                annotations[idx, 6] = bbox[7]           # l1_x
                annotations[idx, 7] = bbox[8]           # l1_y
                annotations[idx, 8] = bbox[10]          # l2_x
                annotations[idx, 9] = bbox[11]          # l2_y
                annotations[idx, 10] = bbox[13]         # l3_x
                annotations[idx, 11] = bbox[14]         # l3_y
                annotations[idx, 12] = bbox[16]         # l4_x
                annotations[idx, 13] = bbox[17]         # l4_y

                if annotations[idx, 4] < 0:
                    annotations[idx, 14] = -1
                else:
                    annotations[idx, 14] = 1
            else:
                annotations[idx, 14] = 1

        if self.transform is not None:
            img, annotations = self.transform(image=img, targets=annotations)

        return img, annotations

def log_dataset(use_artifact, 
        artifact_name, 
        artifact_path, dataset_name, 
        job_type='preprocess dataset', 
        project_name='Content-based RS'):

    run = wandb.init(project=project_name, job_type=job_type)
    run.use_artifact(use_artifact)
    artifact = wandb.Artifact(artifact_name, dataset_name)

    if os.path.isdir(artifact_path):
        artifact.add_dir(artifact_path)
    else:
        artifact.add_file(artifact_path)
    run.log_artifact(artifact)

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []

    for _, (image, target) in enumerate(batch):
        image = torch.from_numpy(image)
        target = torch.from_numpy(target).to(dtype=torch.float)

        imgs.append(image)
        targets.append(target)

    return (torch.stack(imgs, dim=0), targets)