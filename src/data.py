import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class PetsDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        df = pd.read_csv(
            annotations_file,
            sep=" ", comment="#",
            names=["image_id", "class_id", "species", "breed_id"]
        )
        df["path"] = df["image_id"].apply(lambda x: Path(img_dir) / f"{x}.jpg")
        df["label"] = df["class_id"] - 1
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(self.df.loc[idx, "path"]).convert("RGB")
        label = self.df.loc[idx, "label"]
        if self.transform:
            image = self.transform(image)
        return image, label


DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def get_loaders(
    annotations_dir="annotations",
    images_dir="images",
    batch_size=32,
    num_workers=4,
    transform=None,
):
    t = transform or DEFAULT_TRANSFORM

    train_dataset = PetsDataset(f"{annotations_dir}/trainval.txt", images_dir, transform=t)
    test_dataset  = PetsDataset(f"{annotations_dir}/test.txt",     images_dir, transform=t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
