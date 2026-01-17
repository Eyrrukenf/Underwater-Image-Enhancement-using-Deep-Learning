# scripts/train.py

#  Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.datasets.dataset import UnderwaterDataset
from src.models.baseline_cnn import BaselineCNN

#  Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Data loaders
train_dataset = UnderwaterDataset("path_to_train_input", "path_to_train_target", augment=True)
val_dataset   = UnderwaterDataset("path_to_val_input", "path_to_val_target", augment=False)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Model, loss, optimizer
model = BaselineCNN().to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#  Training loop
num_epochs = 20
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for input_img, target_img in train_loader:
        input_img, target_img = input_img.to(device), target_img.to(device)
        optimizer.zero_grad()
        output = model(input_img)
        loss = criterion(output, target_img)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for input_img, target_img in val_loader:
            input_img, target_img = input_img.to(device), target_img.to(device)
            output = model(input_img)
            val_loss += criterion(output, target_img).item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
