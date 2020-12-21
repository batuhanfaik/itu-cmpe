import os
import torch
import torch.nn as nn
from efficient import EfficientNet
import numpy as np
from loader import GenericDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

num_features = EfficientNet.from_name("efficientnet-b0").in_channels
half_in_size = round(num_features / 2)
layer_width = 20
num_classes = 5


class EfficientSpinalNet(nn.Module):
    def __init__(self):
        super(EfficientSpinalNet, self).__init__()

        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(half_in_size, layer_width),
            nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True), )
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(half_in_size + layer_width, layer_width),
            nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True), )
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(half_in_size + layer_width, layer_width),
            nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True), )
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(half_in_size + layer_width, layer_width),
            nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True), )
        self.fc_out = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(layer_width * 4, num_classes), )

    def forward(self, x):
        x1 = self.fc_spinal_layer1(x[:, 0:half_in_size])
        x2 = self.fc_spinal_layer2(torch.cat([x[:, half_in_size:2 * half_in_size], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([x[:, 0:half_in_size], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([x[:, half_in_size:2 * half_in_size], x3], dim=1))

        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)

        x = self.fc_out(x)
        return x


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
DATASET_PATH = "/home/ufuk/cassava-leaf-disease-classification"
model_name = "model_epoch_13.pt"
MODEL_PATH = model_name
BATCH_SIZE = 8


rgb_flag = False




output_path = "output.npy"
target_path = "target.npy"

model = torch.load(MODEL_PATH, map_location=device)
model = model.eval()
model = model.to(device)

dataset_test = GenericDataset(split="train", fold_name="test.txt", path=DATASET_PATH)
loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, unsupervised=True)



out_target = []
out_output = []

for batch_idx, batch in enumerate(loader()):
    img, img_class = batch#new
    img = torch.Tensor(img.float())#new
    img = img.to(device)
    img.requires_grad = False

    img_class = img_class.to(device)
    img_class.requires_grad = False

    output = model(img)
    temp_list = []
    temp_list_class = []

    if not rgb_flag:
        temp_list = torch.cat([output[:, 0:4].sum(dim=1, keepdim=True), output[:, 4:8].sum(dim=1, keepdim=True), output[:, 8:12].sum(dim=1, keepdim=True), output[:, 12:16].sum(dim=1, keepdim=True), 
                    output[:, 16:20].sum(dim=1, keepdim=True)],dim=1)
        temp_list_class = img_class//4
    else:
        temp_list = torch.cat([output[:, 0:6].sum(dim=1, keepdim=True), output[:, 6:12].sum(dim=1, keepdim=True), output[:, 12:18].sum(dim=1, keepdim=True), output[:, 18:24].sum(dim=1, keepdim=True), 
                    output[:, 24:30].sum(dim=1, keepdim=True)],dim=1)
        temp_list_class = img_class//6

    outputs = temp_list
    new_img_class = temp_list_class



    output_np = outputs.data.cpu().numpy()
    target_np = new_img_class.data.cpu().numpy()
    out_output.append(output_np)
    out_target.append(target_np[:, np.newaxis])
    if batch_idx % 20 == 19:
        print("Batch: {}/{}".format(batch_idx + 1, len(loader())))

output_array = np.concatenate(out_output, axis=0)
target_array = np.concatenate(out_target, axis=0)
# np.save(output_path, output_array, allow_pickle=False)
# np.save(target_path, target_array, allow_pickle=False)

tsne = TSNE(n_components=2, init="pca", random_state=0)
print("Starting t-SNE Transform")
output_array = tsne.fit_transform(output_array)
print("Transform complete")

labels = ["", "Bacterial Blight (CBB)", "Brown Streak Disease (CBSD)", "Green Mottle (CGM)",
          "Mosaic Disease (CMD)", "Healthy"]
palette = sns.color_palette("bright", 5)
hue = target_array.flatten()
sns.scatterplot(x=output_array[:, 0], y=output_array[:, 1], hue=hue, palette=palette, legend="full")
# g = sns.relplot(x=output_array[:, 0], y=output_array[:, 1], hue=hue, palette=palette, legend="full", kind="scatter")
title = "t-SNE of EfficientNet-B0 - Rotation Based Label Augmentation with SAM"
plt.title(title)
plt.axis('off')
plt.legend(labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=3)
plt.savefig('{}.png'.format(title), dpi=300, bbox_inches='tight')
print("Plot saved")
