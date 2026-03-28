import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class SmartphoneDSLRDataset(Dataset):
    """
    Unpaired Smartphone <-> DSLR dataset for CycleGAN.

    Domain A: Smartphone photos  (low quality)
    Domain B: DSLR photos        (high quality)

    Images are NOT paired — CycleGAN handles that!
    """

    def __init__(self, root, mode="train", img_size=256):
        self.transform = self._build_transforms(img_size, mode)

        prefix = "train" if mode == "train" else "test"
        self.files_A = sorted(
            self._load_files(os.path.join(root, f"{prefix}A"))
        )
        self.files_B = sorted(
            self._load_files(os.path.join(root, f"{prefix}B"))
        )

        print(f"[Dataset] Mode={mode} | "
              f"Domain A (Smartphone): {len(self.files_A)} images | "
              f"Domain B (DSLR): {len(self.files_B)} images")

    def _load_files(self, folder):
        exts = (".jpg", ".jpeg", ".png", ".webp")
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")
        return [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(exts)
        ]

    def _build_transforms(self, img_size, mode):
        if mode == "train":
            return T.Compose([
                T.Resize(int(img_size * 1.12)),
                T.RandomCrop(img_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5]),
            ])
        else:
            return T.Compose([
                T.Resize(img_size),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, idx):
        img_A = Image.open(
            self.files_A[idx % len(self.files_A)]
        ).convert("RGB")

        img_B = Image.open(
            self.files_B[random.randint(0, len(self.files_B) - 1)]
        ).convert("RGB")

        return {
            "A": self.transform(img_A),
            "B": self.transform(img_B),
        }