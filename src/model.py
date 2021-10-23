import torch
from torch import nn
import timm

class PetfinderModel(nn.Module):

    def __init__(self, model_name, out_features, in_chans, pretrained, num_dense):
        super().__init__()

        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, 128)

        self.fc = nn.Sequential(
                nn.Linear(128+num_dense, 64),
                nn.ReLU(),
                nn.Linear(64, out_features)
                )

        self.dropput = nn.Dropout(0.2)



    def forward(self, img, dense):
        embedding = self.model(img)
        x = self.dropout(embedding)
        x = torch.cat([x, dense], dim=1)
        output = self.fc(x)

        return output
