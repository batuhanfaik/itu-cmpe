import os
import torch
import numpy as np
from loader import GenericDataReader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATASET_PATH = "/mnt/sdb1/datasets/cassava-leaf-disease-classification"
MODEL_PATH = "models/effnet-b0_baseline.pt"
BATCH_SIZE = 32

if not os.path.exists("tsne"):
    os.mkdir("tsne")

output_path = "tsne/output.npy"
target_path = "tsne/target.npy"

model = torch.load(MODEL_PATH, map_location=device)
model = model.eval()
model = model.to(device)

val_loader = torch.utils.data.DataLoader(
    GenericDataReader(mode='val', fold_name="folds/fold_1_val.txt", path=DATASET_PATH),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=1, drop_last=True)

out_target = []
out_output = []

for batch_idx, batch in enumerate(val_loader):
    img = batch['image']
    img_class = batch['label']
    img = img.to(device)
    img.requires_grad = False
    img_class = img_class.to(device)
    img_class.requires_grad = False

    outputs = model(img)
    output_np = outputs.data.cpu().numpy()
    target_np = img_class.data.cpu().numpy()
    out_output.append(output_np)
    out_target.append(target_np[:, np.newaxis])
    if batch_idx % 20 == 19:
        print("Batch: {}/{}".format(batch_idx + 1, len(val_loader)))

output_array = np.concatenate(out_output, axis=0)
target_array = np.concatenate(out_target, axis=0)
np.save(output_path, output_array, allow_pickle=False)
np.save(target_path, target_array, allow_pickle=False)

tsne = TSNE(n_components=2, init="pca", random_state=0)
print("Starting t-SNE Transform")
output_array = tsne.fit_transform(output_array)
print("Transform complete")

labels = ["Bacterial Blight (CBB)", "Brown Streak Disease (CBSD)", "Green Mottle (CGM)", "Mosaic Disease (CMD)", "Healthy"]
palette = sns.color_palette("hls", 5)
hue = target_array.flatten()
sns.scatterplot(x=output_array[:, 0], y=output_array[:, 1], hue=hue, palette=palette,
                legend='full')
title = "t-SNE of EfficientNet-B0"
plt.title(title)
plt.axis('off')
plt.legend(labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=3)
plt.savefig('tsne/{}.png'.format(title), dpi=300, bbox_inches='tight')
print("Plot saved")
