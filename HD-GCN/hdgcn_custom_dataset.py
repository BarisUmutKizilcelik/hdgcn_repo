import torchvision
from PIL import Image
from tqdm import tqdm
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

def build_transform(rescale_size=512, crop_size=448):
    train_transform = torchvision.transforms.Compose([
        # torchvision.transforms.Resize(size=rescale_size),
        torchvision.transforms.RandomHorizontalFlip(),
        # torchvision.transforms.RandomCrop(size=crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=rescale_size),
        torchvision.transforms.CenterCrop(size=crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    train_transform_strong_aug = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=rescale_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=crop_size),
        # RandAugment(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return {'train': train_transform, 'test': test_transform, 'train_strong_aug': train_transform_strong_aug}


class CustomDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, mode, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return 20

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # sample = np.ones((1, 2, 3, 4))
        imarray = np.random.rand(600,600,3) * 255
        sample = Image.fromarray(imarray.astype('uint8')).convert('RGB')
        label = "This is label"

        if self.transform:
            sample = self.transform(sample)

        sample = {'data': sample, 'label': label, 'idx': idx}
        return sample
    
class CLDataTransform(object):
    def __init__(self, transform_weak, transform_strong):
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

    def __call__(self, sample):
        x_w1 = self.transform_weak(sample)
        x_w2 = self.transform_weak(sample)
        x_s = self.transform_strong(sample)
        return x_w1, x_w2, x_s
    
def build_dataset(train_transform, test_transform):
    train_data = CustomDataset('train', transform=train_transform)
    test_data = CustomDataset('test', test_transform)
    return {'train': train_data, 'test': test_data}

def build_hdgcn_dataset_loader():
    transform = build_transform()
    dataset = build_dataset(CLDataTransform(transform['train'], transform['train_strong_aug']), transform['test'])
    train_loader = DataLoader(dataset["train"], batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(dataset["test"], batch_size=16, shuffle=False, num_workers=2, pin_memory=False)
    return dataset, train_loader, test_loader

dataset, train_loader, test_loader = build_hdgcn_dataset_loader()

pbar = tqdm(train_loader, ncols=150, ascii=' >', leave=False, desc='training')
for it, sample in enumerate(pbar):
    index = sample['idx']
    label = sample['label']
    x, x_w, x_s = sample['data']
    assert type(sample['data']) == list and len(sample['data']) == 3
    print("The type of x is: ", sample['data'][0].shape)
    print("The type of x is: ", x[0].shape)
    print("The type of x_w is: ", x_w[0].shape)
    print("The type of x_s is: ", x_s[0].shape)
    print("The label is: ", label[0])
    break

# The type of x is:  torch.Size([16, 3, 600, 600])
# The type of x is:  torch.Size([3, 600, 600])
# The type of x_w is:  torch.Size([3, 600, 600])
# The type of x_s is:  torch.Size([3, 448, 448])
# The label is:  This is label