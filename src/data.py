import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet


TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label


def get_loaders(
    root: str = "data",
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    seed: int = 50,
) -> tuple[DataLoader, DataLoader]:
    trainval = OxfordIIITPet(root=root, split="trainval", target_types="category", transform=None, download=True)
    test     = OxfordIIITPet(root=root, split="test",     target_types="category", transform=None, download=True)

    full_dataset = ConcatDataset([trainval, test])

    total      = len(full_dataset)
    train_size = int(total * train_ratio)
    test_size  = total - train_size

    train_subset, test_subset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_dataset = TransformSubset(train_subset, TRAIN_TRANSFORM)
    test_dataset  = TransformSubset(test_subset,  TEST_TRANSFORM)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_loaders()
    print(f"Train batches : {len(train_loader)}  ({len(train_loader.dataset)} samples)")
    print(f"Test  batches : {len(test_loader)}  ({len(test_loader.dataset)} samples)")

    imgs, labels = next(iter(train_loader))
    print(f"Batch shape   : {imgs.shape}")
    print(f"Labels shape  : {labels.shape}")
