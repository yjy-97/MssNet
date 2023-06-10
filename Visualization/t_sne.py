import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from my_dataset import MyDataSet
from model import MssNet_base as create_model
from utils import read_split_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set seed
torch.manual_seed(3407)
if device == 'cuda':
    torch.cuda.manual_seed(3407)

# set dataset
train_images_path_41 = r"/ad/utils/NC_MCInc/train1"
train_dataset = MyDataSet(train_images_path_41)

train_dataset = MyDataSet(train_images_path_41)
batch_size = 8
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('Using {} dataloader workers every process'.format(nw))
dataloader = torch.utils.data.DataLoader(train_dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         pin_memory=True,
                                         num_workers=nw,
                                         collate_fn=train_dataset.collate_fn)

net = create_model(num_classes=2).to(device)

weights_dict = torch.load(r'best_model_AD_NC_s3.pth', map_location=device)
for k in list(weights_dict.keys()):
    if "head" in k:
        del weights_dict[k]
net.load_state_dict(weights_dict, strict=False)



def gen_features():
    net.eval()
    targets_list = []
    outputs_list = []

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            targets_np = targets.data.cpu().numpy()

            outputs = net(inputs)
            outputs_np = outputs.data.cpu().numpy()

            targets_list.append(targets_np[:, np.newaxis])
            outputs_list.append(outputs_np)

            if ((idx + 1) % 10 == 0) or (idx + 1 == len(dataloader)):
                print(idx + 1, '/', len(dataloader))

    targets = np.concatenate(targets_list, axis=0)
    outputs = np.concatenate(outputs_list, axis=0).astype(np.float64)

    return targets, outputs


def tsne_plot(save_dir, targets, outputs):
    print('generating t-SNE plot...')
    # tsne_output = bh_sne(outputs)
    tsne = TSNE(random_state=0)
    tsne_output = tsne.fit_transform(outputs)

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df["lab"] = targets

    data_label = []
    for v in df.lab.tolist():
        if v == 1:
            data_label.append("HC")
        else:
            data_label.append("AD")

    df['Subjects'] = data_label

    plt.rcParams['figure.figsize'] = 10, 10

    markers = {"AD": "v", "HC": "^"}
    sns.scatterplot(x='x', y='y', hue='Subjects', markers=markers,
                    palette=sns.color_palette("Set2", 2),
                    data=df)

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    plt.savefig(os.path.join(save_dir, 'tsne.png'), bbox_inches='tight')
    print('done!')


targets, outputs = gen_features()
tsne_plot(r'/tp', targets, outputs)
