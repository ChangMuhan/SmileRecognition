import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class SmileDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 包含 'smile' 和 'non_smile' 子目录的根目录.
            transform (callable, optional): 应用于图像的转换.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 遍历 'smile' 和 'non_smile' 目录
        for class_name in ['smile', 'non_smile']:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                self.image_paths.append(image_path)
                # label 0: 'non_smile', label 1: 'smile'
                self.labels.append(1 if class_name == 'smile' else 0)

        self.len = len(self.image_paths)
        if self.len != len(self.labels):
            raise ValueError("The number of image paths and labels does not match.")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

if __name__ == '__main__':
    # 定义数据转换
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整大小以适应 ResNet-18 的输入
        transforms.ToTensor(),           # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    root_dir = 'dataset/train'
    smile_dataset = SmileDataset(root_dir=root_dir, transform=data_transforms)

    # 创建 DataLoader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(smile_dataset, batch_size=32, shuffle=True)

    # 迭代一个 batch
    for images, labels in dataloader:
        print('Image batch shape:', images.shape)