import torch
import torch.nn as nn


class PneumoniaClassificationNet(nn.Module):
    def __init__(self, image_channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=32, padding='same', kernel_size=3),  # 32 x 256 x 256 224
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 x 128 x 112
            nn.Dropout(0.2),

            nn.Conv2d(in_channels=32, out_channels=64, padding='same', kernel_size=3),  # 64 x 128 x 128
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 x 64 x 64
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=64, out_channels=128, padding='same', kernel_size=3),  # 128 x 64 x 64
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128 x 32 x 32
            nn.Dropout(0.3),

            nn.Conv2d(in_channels=128, out_channels=256, padding='same', kernel_size=3),  # 256 x 32 x 32
            nn.ReLU(),
            nn.MaxPool2d(2),  # 256 x 16 x 16
            nn.Dropout(0.2),

            nn.Conv2d(in_channels=256, out_channels=256, stride=2, kernel_size=3),  # 256 x 7 x 7
            nn.ReLU(),
            nn.MaxPool2d(2),  # 256 x 3 x 3
            nn.Dropout(0.2)
        )

        self.head = nn.Sequential(
            nn.Linear(in_features=2304, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=2)
        )

    def forward(self, x):
        out = self.encoder(x)
        out = out.view(x.size(0), -1)
        out = self.head(out)
        return out

    def predict_proba(self, x):
        result = self.forward(x)
        return nn.Softmax(dim=1)(result)


def pneumoniaNet(pretrained=False):
    model = PneumoniaClassificationNet()
    if pretrained:
        params = torch.load('pneumaniaModule/weightPneumaniaModel.pt')
        model.load_state_dict(params)
    return model
