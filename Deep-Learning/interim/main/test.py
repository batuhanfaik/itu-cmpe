import torch
import torch.nn as nn
from efficientnet import EfficientNet
from loader import GenericDataReader, DataReader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATASET_PATH = "/home/ufuk/cassava-leaf-disease-classification"
MODEL_PATH = "results/best_models/effnet-b0_spinal.pt"
BATCH_SIZE = 8

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


model = torch.load(MODEL_PATH, map_location=device)
model = model.eval()
model = model.to(device)

loader = torch.utils.data.DataLoader(
    GenericDataReader(mode='test', fold_name="test.txt", path=DATASET_PATH),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=1, drop_last=True)

total_true = 0
total_false = 0

for i, batch in enumerate(loader):
    img = batch['image']
    img_class = batch['label']
    img = img.to(device)
    img.requires_grad = False

    img_class = img_class.to(device)
    img_class.requires_grad = False
    output = model(img)

    _, prediction = torch.max(output.data, 1)

    total_true += torch.sum(prediction == img_class.data)
    total_false += torch.sum(prediction != img_class.data)

acc = total_true.item() * 1.0 / (total_true.item() + total_false.item())
print("Test scores:")
print("Accuracy %f" % acc)
