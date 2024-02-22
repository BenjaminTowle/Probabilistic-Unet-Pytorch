import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from load_LIDC_data import LIDC_IDRI
from probabilistic_unet import ProbabilisticUnet

torch.set_grad_enabled(False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = LIDC_IDRI(dataset_location = 'data/')
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.1 * dataset_size))
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(dataset, batch_size=1, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler)
print("Number of training/test patches:", (len(train_indices),len(test_indices)))

net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=2, no_convs_fcomb=4, beta=10.0)
net.to(device)
net.load_state_dict(torch.load('model.pth'))
net.eval()

import matplotlib.pyplot as plt
n_samples = 3

for step, (patch, mask, _) in enumerate(test_loader): 
    patch = patch.to(device)
    mask = mask.to(device)
    mask = torch.unsqueeze(mask,1)
    net.forward(patch, mask, training=False)
    samples = []
    for _ in range(n_samples):
        sample = net.sample(testing=True)
        samples.append(sample.squeeze().cpu().detach().numpy() > 0.0)

    fig, ax = plt.subplots(1, n_samples+2, figsize=(15,5))
    ax[0].imshow(patch.squeeze().cpu().detach().numpy(), cmap='gray')
    ax[0].set_title('Input patch')
    for i in range(n_samples):
        ax[i+1].imshow(samples[i], cmap='gray')
        ax[i+1].set_title(f'Sample {i+1}')
    ax[-1].imshow(mask.squeeze().cpu().detach().numpy(), cmap='gray')
    plt.show()

        