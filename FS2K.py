import os
from PIL import Image
import config as cfg
import json
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils import data

def load_data(json_path, attributes, batch_size, num_workers, mode):
    dataset = FS2K(json_path, attributes)

    if mode == 'train':
        shuffle = True
    else:
        shuffle = False

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers)
    return dataloader

class FS2K(data.Dataset):
    def __init__(self, json_path, attributes):
        self.img_paths = list()
        self.labels = list()
        
        transform = [transforms.Resize(size=(224, 224)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])]
        self.transform = transforms.Compose(transform)

        with open(os.path.join(json_path), 'r') as f:
            annotations = json.load(f)
        
        for anno in annotations:
            img_name = anno['image_name'].replace('photo', 'sketch').replace('image', 'sketch')
            if '2' in img_name.split('/')[0]:
                img_path = img_name + '.png'
            else:
                img_path = img_name + '.jpg'
            self.img_paths.append(img_path)
            label = list()
            for attr in attributes:
                label.append(anno[attr])
            self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.labels[index]
        image = Image.open(os.path.join(cfg.root, img_path)).convert('RGB')
        image = self.transform(image)
        return image, label



