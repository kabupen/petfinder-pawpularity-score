import pandas as pd
import cv2

from torch.utils.data  import Dataset
import albumentations

class PetfinderDataset(Dataset):

    def __init__(self,
                 image_id : pd.DataFrame,
                 dense_features : pd.DataFrame,
                 targets,
                 transform=None):

        self.image_path = list((IMG_PATH + image_id + '.jpg').to_numpy())
        self.dense_features = dense_features.to_numpy()
        self.targets = targets
        self.transform = transform
    

    def __len__(self):
        
        return len(self.image_path)


    def __getitem__(self, idx):

        path = self.image_path[idx]

        # Load images 
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # augmentations
        if self.transform is not None:
            img = self.transform(img)


        dense = self.dense_features[idx, :]
        label = torch.tensor(self.targets[idx]).float()

        return img, dense, label

