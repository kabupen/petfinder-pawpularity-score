import albumentations

def get_transform(size):
    return albumentations.Compose([
        albumentations.Resize(size, size),
        albumentations.Normalize(
                mean = [0.485, 0.456, 0.486],
                std = [0.229, 0.224, 0.225],
                ),
        ToTensorV2(p=1.0)
        ])
