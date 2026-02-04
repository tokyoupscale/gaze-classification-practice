import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

from pathlib import Path


dataset_path = Path("data/eye_tracking")
BATCH_SIZE = 128 # по 128 картинок в батч сучкаа


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]) # теперь изображение это тензор

final_dataset = ImageFolder(root=dataset_path, transform=transform)

# проверка что все нормик
# print(final_dataset.classes)
# print(final_dataset.class_to_idx)
# print(final_dataset.samples[:3])
# print(len(final_dataset))

train_size = int(0.75 * len(final_dataset))
val_size = int(0.10 * len(final_dataset))
test_size = len(final_dataset) - train_size - val_size

# создаем три сабсета под каждую выборку
train_dataset, val_dataset, test_dataset = random_split(
    final_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
) 

# print(f"train {len(train_dataset)} | val {len(val_dataset)} | test {len(test_dataset)}")

# хочу воркеров 
def get_loaders():
    train_loader = DataLoader(
        train_dataset,
        shuffle=True, # для обучения лучше перемешать
        batch_size=BATCH_SIZE,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader

# print(final_dataset.class_to_idx)