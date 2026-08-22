from pathlib import Path
import json
import copy
import numpy as np
import pandas as pd
from PIL import Image
 
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
 
ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
SAMPLES = ROOT / "data" / "sample_images"
DATA_ROOT = ROOT / "data" / "fashion_mnist"
for p in [MODELS, RESULTS, SAMPLES, DATA_ROOT]: p.mkdir(parents=True, exist_ok=True)
 
SEED = 42
BATCH_SIZE = 128
HEAD_EPOCHS = 3
FINETUNE_EPOCHS = 2
HEAD_LR = 1e-3
FINETUNE_LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)
 
class_names = [
    "T-shirt_top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle_boot"
]
 
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
 
# Train/validation from official 60,000 train split. Test untouched until final evaluation.
full_train = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=transform)
test_ds = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=transform)
labels = np.array(full_train.targets)
indices = np.arange(len(full_train))
train_idx, val_idx = train_test_split(
    indices, test_size=5000, stratify=labels, random_state=SEED
)
train_ds = Subset(full_train, train_idx)
val_ds = Subset(full_train, val_idx)
 
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print("Split sizes:", len(train_ds), len(val_ds), len(test_ds))
 
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
for p in model.parameters():
    p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(DEVICE)
 
criterion = nn.CrossEntropyLoss()
 
def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        if training: optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(xb)
            loss = criterion(logits, yb)
            if training:
                loss.backward(); optimizer.step()
        total_loss += loss.item() * yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return total_loss / total, correct / total
 
def train_phase(model, train_loader, val_loader, optimizer, epochs):
    best_state = copy.deepcopy(model.state_dict())
    best_val = -1.0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader)
        print(f"epoch={epoch} train_acc={tr_acc:.4f} val_acc={va_acc:.4f}")
        if va_acc > best_val:
            best_val = va_acc
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return best_val
 
# Phase 1: feature extraction / head training
optimizer = torch.optim.Adam(model.fc.parameters(), lr=HEAD_LR)
feature_val_acc = train_phase(model, train_loader, val_loader, optimizer, HEAD_EPOCHS)
print("Feature-extraction validation accuracy:", feature_val_acc)
 
finetuned = False
final_val_acc = feature_val_acc
if feature_val_acc < 0.80:
    finetuned = True
    # Unfreeze only the late layer4 block + classifier head.
    for p in model.layer4.parameters(): p.requires_grad = True
    for p in model.fc.parameters(): p.requires_grad = True
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=FINETUNE_LR
    )
    final_val_acc = train_phase(model, train_loader, val_loader, optimizer, FINETUNE_EPOCHS)
    print("Fine-tuned validation accuracy:", final_val_acc)
 
# Final test evaluation: test split is used here for the first time.
model.eval()
y_true, y_pred, confidences = [], [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        probs = torch.softmax(model(xb), dim=1)
        conf, pred = probs.max(dim=1)
        y_true.extend(yb.numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
        confidences.extend(conf.cpu().numpy().tolist())
 
test_acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
print("Test accuracy:", test_acc)
print("Confusion matrix:\n", cm)
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
 
pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(RESULTS / "confusion_matrix.csv")
pd.DataFrame(report).T.to_csv(RESULTS / "classification_report.csv")
 
# Identify largest off-diagonal confusion pairs from real predictions.
cm_off = cm.copy(); np.fill_diagonal(cm_off, 0)
pairs = []
for _ in range(2):
    i, j = np.unravel_index(np.argmax(cm_off), cm_off.shape)
    pairs.append({"true": class_names[i], "predicted": class_names[j], "count": int(cm_off[i, j])})
    cm_off[i, j] = 0
print("Top confusion directions:", pairs)
 
# Save model weights and metadata needed to reconstruct architecture.
torch.save(model.state_dict(), MODELS / "product_classifier.pt")
meta = {
    "architecture": "resnet18",
    "input_size": [224, 224],
    "class_names": class_names,
    "batch_size": BATCH_SIZE,
    "optimizer": "Adam",
    "head_lr": HEAD_LR,
    "finetune_lr": FINETUNE_LR,
    "head_epochs": HEAD_EPOCHS,
    "finetune_epochs": FINETUNE_EPOCHS if finetuned else 0,
    "feature_extraction_val_accuracy": feature_val_acc,
    "final_val_accuracy": final_val_acc,
    "fine_tuning_required": finetuned,
    "test_accuracy": test_acc,
    "top_confusions": pairs,
}
with open(MODELS / "product_classifier_meta.json", "w") as f: json.dump(meta, f, indent=2)
with open(RESULTS / "part2_metrics.json", "w") as f: json.dump(meta, f, indent=2)
 
# Export at least five REAL test-split PNGs with true labels in filenames.
raw_test = datasets.FashionMNIST(DATA_ROOT, train=False, download=False)
chosen = []
seen = set()
for idx, (img, label) in enumerate(raw_test):
    if label not in seen:
        safe = class_names[label].lower()
        path = SAMPLES / f"{idx:05d}_{safe}.png"
        img.save(path)
        chosen.append(str(path.relative_to(ROOT)))
        seen.add(label)
    if len(chosen) >= 5: break
print("Exported sample images:", chosen)
print("Saved model:", MODELS / "product_classifier.pt")