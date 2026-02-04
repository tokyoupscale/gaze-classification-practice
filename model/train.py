import torch
from torch import nn
import torch.optim as optim
from data.loaders import get_loaders
from model.model import model
from tqdm import tqdm

def train():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"using {device}")

    model_instance = model.to(device)
    print(f"Model on device: {next(model_instance.parameters()).device}")

    train_loader, val_loader, test_loader = get_loaders()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_instance.parameters(), lr=0.0001)

    EPOCHS = 10

    for epoch in tqdm(range(EPOCHS), desc=f"epoch processing"):
        model_instance.train()

        for batch_idx, (images, labels) in tqdm(enumerate(train_loader), desc="batch processing"):
            images, labels = images.to(device), labels.to(device)

            outputs = model_instance(images)

            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # if batch_idx % 50 == 0:
            #     print(f"epoch {epoch+1}/{EPOCHS} | batch {batch_idx}/{len(train_loader)} | loss {loss.item():.4f}")

            model_instance.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)

                    outputs = model_instance(images)

                    _, predicted = torch.max(outputs, 1)

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f"\nepoch {epoch+1} accuracy: {accuracy:.2f}%")

    torch.save(model_instance.state_dict(), "gaze-classification.pth")
    print("done")

if __name__ == '__main__':
    train()