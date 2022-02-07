import os

from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, src_dir, label_dir, transform_resize, target_transform_resize, transform_crop, target_transform_crop):
        self.img_src = sorted([item for item in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, item))])
        self.img_labels = sorted([item for item in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, item))])
        self.src_dir = src_dir
        self.label_dir = label_dir
        self.transform_resize = transform_resize
        self.target_transform_resize = target_transform_resize
        self.transform_crop = transform_crop
        self.target_transform_crop = target_transform_crop

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        src_path = os.path.join(self.src_dir, self.img_src[idx])
        label_path = os.path.join(self.label_dir, self.img_labels[idx])

        image_src = Image.open(src_path)
        image_label = Image.open(label_path)

        width, height = image_src.size

        if width > 640 and height > 640:
            if self.transform_crop:
                image_src = self.transform_crop(image_src)
            if self.target_transform_crop:
                image_label = self.target_transform_crop(image_label)
        else:
            if self.transform_resize:
                image_src = self.transform_resize(image_src)
            if self.target_transform_resize:
                image_label = self.target_transform_resize(image_label)
        
        return image_src, image_label


def get_data():
    transform_resize = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])
    transform_label_resize = transforms.Compose([transforms.Grayscale(), transforms.Resize((640, 640)), transforms.ToTensor()])

    transform_crop = transforms.Compose([transforms.CenterCrop((640, 640)), transforms.ToTensor()])
    transform_label_crop = transforms.Compose([transforms.Grayscale(), transforms.CenterCrop((640, 640)), transforms.ToTensor()])
    
    data = MyDataset(src_dir='./data/AUTOENCODER/augmented_src', 
                     label_dir='./data/AUTOENCODER/augmented_labels', 
                     transform_resize=transform_resize,
                     target_transform_resize=transform_label_resize,
                     transform_crop=transform_crop,
                     target_transform_crop=transform_label_crop)

    train_len = int(0.7 * len(data))
    val_len = int(0.2 * len(data))
    test_len = int(len(data) - (train_len + val_len))

    data_train, data_val, data_test = random_split(data,
                                      [train_len, val_len, test_len])

    return data_train, data_val, data_test
