from models import CNN
from data import ds, get_fold_splits
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

num_epochs = 1

def train_with_folds(train_folds, val_folds, test_folds):
    """Train model with specified fold configuration."""
    print(f"\nTraining with folds - Train: {train_folds}, Val: {val_folds}, Test: {test_folds}")
    
    # Get fold splits
    train_indices, val_indices, test_indices = get_fold_splits(ds, train_folds, val_folds, test_folds)
    
    # Create dataloaders
    train_loader = DataLoader(Subset(ds, train_indices), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(ds, val_indices), batch_size=32, shuffle=False)
    test_loader = DataLoader(Subset(ds, test_indices), batch_size=32, shuffle=False)
    
    # Initialize model
    model = CNN(in_channels=1, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            data = batch['audio']
            targets = batch['label']
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                data = batch['audio']
                targets = batch['label']
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")
    
    # Test
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch in test_loader:
            data = batch['audio']
            targets = batch['label']
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            test_total += targets.size(0)
            test_correct += (predicted == targets).sum().item()
    
    test_acc = 100 * test_correct / test_total
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    return test_acc

# Run different fold configurations
random.seed(42)
used_test_folds = set()
all_folds = list(range(1, 11))

for i in range(3):  # Run 3 configurations
    # Pick random test fold that hasn't been used
    available_test_folds = [f for f in all_folds if f not in used_test_folds]
    test_fold = random.choice(available_test_folds)
    used_test_folds.add(test_fold)
    
    # Pick random val fold (different from test fold)
    available_val_folds = [f for f in all_folds if f != test_fold]
    val_fold = random.choice(available_val_folds)
    
    # Remaining folds for training
    train_folds = [f for f in all_folds if f not in [test_fold, val_fold]]
    
    print(f"\n{'='*50}")
    print(f"CONFIGURATION {i+1}")
    print(f"{'='*50}")
    train_with_folds(train_folds, [val_fold], [test_fold])