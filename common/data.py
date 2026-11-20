"""
Data loading utilities for CIFAR-10 / CIFAR-100.

* Standard CIFAR augmentation (random crop + horizontal flip) for training.
* Optional train/validation split (used by NAS so that the test set is never
  touched during the search).
* Optional sub-sampling (used by NAS proxy training to keep each candidate cheap).
* `fake=True` swaps the real dataset for torchvision.FakeData so the whole
  pipeline can be smoke-tested without downloading CIFAR.
"""
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

# Per-channel statistics computed on the CIFAR training sets
STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


def get_transforms(dataset, train=True):
    mean, std = STATS[dataset]
    if train:
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([T.ToTensor(), T.Normalize(mean, std)])


def _build_dataset(dataset, root, train, transform, fake, fake_size):
    if fake:
        n = fake_size if train else max(fake_size // 5, 64)
        return torchvision.datasets.FakeData(
            size=n, image_size=(3, 32, 32), num_classes=NUM_CLASSES[dataset],
            transform=transform, random_offset=0 if train else 10_000)
    cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
    return cls(root=root, train=train, download=True, transform=transform)


def get_loaders(dataset="cifar100", root="./data", batch_size=128, workers=4,
                val_size=0, train_subset=None, seed=42, fake=False, fake_size=512):
    """Return (train_loader, val_loader, test_loader).

    val_size > 0 carves a held-out validation set out of the *training* split
    (with test-time transforms). If val_size == 0, val_loader is None.
    train_subset limits the number of training images (NAS proxy training).
    """
    train_full = _build_dataset(dataset, root, True, get_transforms(dataset, True), fake, fake_size)
    train_eval = _build_dataset(dataset, root, True, get_transforms(dataset, False), fake, fake_size)
    test_set = _build_dataset(dataset, root, False, get_transforms(dataset, False), fake, fake_size)

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(train_full), generator=g).tolist()
    if val_size > 0:
        val_size = min(val_size, len(idx) // 5) if fake else val_size
        val_idx, train_idx = idx[:val_size], idx[val_size:]
    else:
        val_idx, train_idx = [], idx
    if train_subset:
        train_idx = train_idx[:train_subset]

    pin = torch.cuda.is_available()
    kw = dict(num_workers=workers, pin_memory=pin, persistent_workers=workers > 0)
    train_loader = DataLoader(Subset(train_full, train_idx), batch_size=batch_size,
                              shuffle=True, drop_last=True, **kw)
    val_loader = (DataLoader(Subset(train_eval, val_idx), batch_size=256, shuffle=False, **kw)
                  if val_idx else None)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, **kw)
    return train_loader, val_loader, test_loader
