import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import yaml

with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(batch_labels.cpu().numpy())
    test_f1 = f1_score(test_labels, test_preds, average='binary')
    return test_f1