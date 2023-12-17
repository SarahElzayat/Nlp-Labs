import torch
import numpy as np  

from torch.utils.data import Dataset, DataLoader
x= torch.rand(4,4)

y=x.view(-1,8)

b= x.numpy()

if torch.cuda.is_available():
    device=torch.device("cuda")
    x=torch.ones(5,device=device)
    y=torch.ones(5)
    y=y.to(device)
    z=x+y
    print(z)
    print(z.to("cpu",torch.double))
else:
    print("cuda is not available")
    
class wineDataset(Dataset):
    def __init__(self):
        xy=np.loadtxt('./data/wine.csv',delimiter=',',dtype=np.float32,skiprows=1)
        self.x=torch.from_numpy(xy[:,1:])
        self.y=torch.from_numpy(xy[:,[0]])
        self.n_samples=xy.shape[0]
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    def __len__(self):
        return self.n_samples
dataset=wineDataset()
first_data=dataset[0]
features,labels=first_data
print(features,labels)