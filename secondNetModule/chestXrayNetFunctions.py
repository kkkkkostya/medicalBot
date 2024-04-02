import torch
import torch.nn as nn


class ChestXrayNet(nn.Module):
    def __init__(self, image_channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=32, padding='same', kernel_size=3),  # 32 x 1024 x 1024
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 x 512 x 512
            nn.Dropout(0.2),

            nn.Conv2d(in_channels=32, out_channels=64, padding='same', kernel_size=3),  # 64 x 512 x 512
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 x 256 x 256
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=64, out_channels=128, padding='same', kernel_size=3),  # 128 x 256 x 256
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128 x 128 x 128
            nn.Dropout(0.3),

            nn.Conv2d(in_channels=128, out_channels=256, padding='same', kernel_size=3),  # 256 x 128 x 128
            nn.ReLU(),
            nn.MaxPool2d(2),  # 256 x 64 x 64
            nn.Dropout(0.2),

            nn.Conv2d(in_channels=256, out_channels=256, stride=2, kernel_size=2),  # 256 x 32 x 32
            nn.ReLU(),
            nn.MaxPool2d(2),  # 256 x 16 x 16
            nn.Dropout(0.2),

            nn.Conv2d(in_channels=256, out_channels=512, padding='same', kernel_size=3),  # 512 x 16 x 16
            nn.ReLU(),
            nn.MaxPool2d(2),  # 512 x 8 x 8
            nn.Dropout(0.2),

            nn.Conv2d(in_channels=512, out_channels=512, stride=2, kernel_size=2),  # 512 4 x 4
            nn.ReLU(),
            nn.MaxPool2d(2),  # 512 x 2 x 2
            nn.Dropout(0.2),

            nn.Flatten()  # 2048
        )

        self.head = nn.Sequential(
            nn.Linear(in_features=2048, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=14),
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.head(out)
        return out

    def predict_proba(self, x):
        result = self.forward(x)
        return nn.Sigmoid()(result)


def chestXrayNet(pretrained=False):
    model = ChestXrayNet()
    if pretrained:
        params = torch.load('secondNetModule/chest_X14_weights.pt',map_location=torch.device('cpu'))
        model.load_state_dict(params)
    return model

