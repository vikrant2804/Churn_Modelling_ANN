import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from model import MyNN
from utils import mixup_data
from sklearn.metrics import f1_score

# Load configuration
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(train_dataset, val_dataset, input_dim):
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # Model initialization
    model = MyNN(input_dim=input_dim, output_dim=config['model']['output_dim'],
                 num_hidden_layers=config['model']['num_hidden_layers'],
                 neurons_per_layer=config['model']['neurons_per_layer'],
                 dropout_rate=config['model']['dropout_rate']).to(device)

    # Optimizer and loss
    optimizer = getattr(optim, config['training']['optimizer'])(
        model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay']
    )
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0], dtype=torch.float).to(device))

    # Training loop with early stopping
    best_val_f1 = 0.0
    patience_counter = 0
    best_model_state = None

    for epoch in range(config['training']['epochs']):
        model.train()
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            batch_features, labels_a, labels_b, lam = mixup_data(batch_features, batch_labels, config['training']['alpha'])

            # Forward and backward pass
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = model(batch_features)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(batch_labels.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds, average='binary')
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= config['training']['patience']:
                break

    model.load_state_dict(best_model_state)
    return model