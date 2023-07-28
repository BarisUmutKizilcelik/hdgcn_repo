import pickle
import numpy as np
import torch.utils.data as Data
from PIL import Image
import tools
import torch
from random import choice
import random 

class mnist_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, dataset='mnist', noise_type='symmetric', noise_rate=0.5, split_per=0.9, random_seed=1, num_class=10):
            
        self.transform = transform
        self.target_transform = target_transform
        self.train = train 
        original_images = np.load('data/mnist/train_images.npy')
        original_labels = np.load('data/mnist/train_labels.npy')

        # clean images and noisy labels (training and validation)
        self.train_data, self.val_data, self.train_labels, self.val_labels = tools.dataset_split(original_images, 
                                                                             original_labels, dataset, noise_type, noise_rate, split_per, random_seed, num_class)

       

    def __getitem__(self, index):
           
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
        else:
            img, label = self.val_data[index], self.val_labels[index]
            
        img = Image.fromarray(img)
           
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label, index
    def __len__(self):
            
        if self.train:
            return len(self.train_data)
   
        else:
            return len(self.val_data)
 
class mnist_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):
            
        self.transform = transform
        self.target_transform = target_transform
        
        self.test_data = np.load('data/mnist/test_images.npy')
        self.test_labels = np.load('data/mnist/test_labels.npy') - 1 # 0-9
        
    def __getitem__(self, index):
        
        img, label = self.test_data[index], self.test_labels[index]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label, index
    
    def __len__(self):
        return len(self.test_data)
  
    
class cifar10_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, dataset='cifar10', noise_type='symmetric', noise_rate=0.5, split_per=0.9, random_seed=1, num_class=10):
            
        self.transform = transform
        self.target_transform = target_transform
        self.train = train 
        
        original_images = np.load('data/cifar10/train_images.npy')
        original_labels = np.load('data/cifar10/train_labels.npy')
        

        # clean images and noisy labels (training and validation)
        self.train_data, self.val_data, self.train_labels, self.val_labels = tools.dataset_split(original_images, 
                                                                             original_labels, dataset, noise_type, noise_rate, split_per, random_seed, num_class)



        

        if self.train:      
            self.train_data = self.train_data.reshape((-1, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))
        
        else:
            self.val_data = self.val_data.reshape((-1, 3, 32, 32))
            self.val_data = self.val_data.transpose((0, 2, 3, 1))
        
    def __getitem__(self, index):
           
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
            
        else:
            img, label = self.val_data[index], self.val_labels[index]
            
        img = Image.fromarray(img)
           
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label, index
    def __len__(self):
            
        if self.train:
            return len(self.train_data)
        
        else:
            return len(self.val_data)
        
class cifar10_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):
            
        self.transform = transform
        self.target_transform = target_transform
           
        self.test_data = np.load('data/cifar10/test_images.npy')
        self.test_labels = np.load('data/cifar10/test_labels.npy')
        self.test_data = self.test_data.reshape((-1, 3, 32, 32))
        self.test_data = self.test_data.transpose((0, 2, 3, 1)) 
    def __getitem__(self, index):
        
        img, label = self.test_data[index], self.test_labels[index]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label, index
    
    def __len__(self):
        return len(self.test_data)
    


    
class cifar100_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, dataset='cifar100', noise_type='symmetric', noise_rate=0.5, split_per=0.9, random_seed=1, num_class=100):
            
        self.transform = transform
        self.target_transform = target_transform
        self.train = train 
        
        original_images = np.load('data/cifar100/train_images.npy')
        original_labels = np.load('data/cifar100/train_labels.npy')

        # clean images and noisy labels (training and validation)
        self.train_data, self.val_data, self.train_labels, self.val_labels = tools.dataset_split(original_images, 
                                                                             original_labels, dataset, noise_type, noise_rate, split_per, random_seed, num_class)



        if self.train:      
            self.train_data = self.train_data.reshape((-1, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1)) 
        
        else:
            self.val_data = self.val_data.reshape((-1, 3, 32, 32))
            self.val_data = self.val_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
           
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
            
        else:
            img, label = self.val_data[index], self.val_labels[index]
            
        img = Image.fromarray(img)
           
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label, index
    def __len__(self):
            
        if self.train:
            return len(self.train_data)
        
        else:
            return len(self.val_data)
        
        
class cifar100_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):
            
        self.transform = transform
        self.target_transform = target_transform
           
        self.test_data = np.load('data/cifar100/test_images.npy')
        self.test_labels = np.load('data/cifar100/test_labels.npy')
        self.test_data = self.test_data.reshape((-1, 3, 32, 32))
        self.test_data = self.test_data.transpose((0, 2, 3, 1)) 

    def __getitem__(self, index):
        
        img, label = self.test_data[index], self.test_labels[index]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label, index
    
    def __len__(self):
        return len(self.test_data)

          
class fmnist_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, dataset='fmnist', noise_type='symmetric', noise_rate=0.5, split_per=0.9, random_seed=1, num_class=10):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        original_images = np.load('data/fashionmnist/train_images.npy').reshape((-1, 1, 28, 28))
        original_labels = np.load('data/fashionmnist/train_labels.npy')
    

        self.train_data, self.val_data, self.train_labels, self.val_labels = tools.dataset_split(original_images, 
                                                                             original_labels, dataset, noise_type, noise_rate, split_per, random_seed, num_class)
        

    def __getitem__(self, index):

        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
        else:
            img, label = self.val_data[index], self.val_labels[index]
        img = torch.from_numpy(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):

        if self.train:
            return len(self.train_data)

        else:
            return len(self.val_data)

        
        

class fmnist_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.test_data = np.load('data/fashionmnist/test_images.npy').reshape((-1, 1, 28, 28))
        self.test_labels = np.load('data/fashionmnist/test_labels.npy')

    def __getitem__(self, index):
        img, label = self.test_data[index], self.test_labels[index]
        img = torch.from_numpy(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.test_data)
    
    
class ntu_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, dataset='ntu60', noise_type='symmetric', noise_rate=0.5, split_per=0.9, random_seed=1, num_class=60):
            
        self.transform = transform
        self.target_transform = target_transform
        self.train = train 
        
        npz_data = np.load('/cvhci/temp/prgc/hdgcn_filtered/NTU60_CV.npz')
        original_images = npz_data['x_train'] 
        original_labels = np.where(npz_data['y_train'] > 0)[1]
        
        # original_images = np.load('/cvhci/temp/prgc/hdgcn_filtered/gen_data/xview/train_data.npy')
        # self.label_path = '/cvhci/temp/prgc/hdgcn_filtered/gen_data/xview/train_label.pkl'
        # with open(self.label_path, 'rb') as f:
        #     self.sample_name, self.label = pickle.load(f)
        # original_labels = self.label
        

        # clean images and noisy labels (training and validation)
        self.train_data, self.val_data, self.train_labels, self.val_labels = tools.dataset_split(original_images, 
                                                                             original_labels, dataset, noise_type, noise_rate, split_per, random_seed, num_class)



        
        N, T, _ = self.train_data.shape
        self.data = self.train_data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
        # if self.train:      
        #     self.train_data = self.train_data.reshape((-1, 3, 32, 32))
        #     self.train_data = self.train_data.transpose((0, 2, 3, 1))
        
        # else:
        #     self.val_data = self.val_data.reshape((-1, 3, 32, 32))
        #     self.val_data = self.val_data.transpose((0, 2, 3, 1))
        
    def __getitem__(self, index):
           
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
            
        else:
            img, label = self.val_data[index], self.val_labels[index]
            
        img = Image.fromarray(img)
           
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label, index
    def __len__(self):
            
        if self.train:
            return len(self.train_data)
        
        else:
            return len(self.val_data)
        
        
class ntu_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):
            
        self.transform = transform
        self.target_transform = target_transform
        
        npz_data = np.load('/cvhci/temp/prgc/hdgcn_filtered/NTU60_CV.npz')
        self.test_data = npz_data['x_test']
        self.test_labels = np.where(npz_data['y_test'] > 0)[1]
        
        N, T, _ = self.test_data.shape
        self.data = self.test_data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
        print(self.test_data.shape)
        # self.test_data = np.load('data/cifar10/test_images.npy')
        # self.test_labels = np.load('data/cifar10/test_labels.npy')
        # self.test_data = self.test_data.reshape((-1, 3, 32, 32))
        # self.test_data = self.test_data.transpose((0, 2, 3, 1)) 
    def __getitem__(self, index):
        
        img, label = self.test_data[index], self.test_labels[index]
        
        img = Image.fromarray(img)
        
  
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
            
        
     
        return img, label, index
    
    def __len__(self):
        return len(self.test_data)
    