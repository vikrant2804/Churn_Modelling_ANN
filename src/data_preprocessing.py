import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import torch
from torch.utils.data import TensorDataset

# Load configuration
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def load_data():
    df = pd.read_csv(config['data']['raw_path'])
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    return df

def preprocess_data(df):
    # Encode categorical variables
    df = pd.get_dummies(df, columns=['Geography'], prefix='Geography', drop_first=True)
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])

    # Create new features
    df['Balance_to_Salary'] = df['Balance'] / df['EstimatedSalary'].replace(0, 1e-6)
    df['Age_Tenure'] = df['Age'] * df['Tenure']
    df['NumOfProducts_Binned'] = pd.cut(df['NumOfProducts'], bins=[0, 2, 4], labels=['Low', 'High'])
    df = pd.get_dummies(df, columns=['NumOfProducts_Binned'], prefix='NumOfProducts')
    df['CreditScore_Binned'] = pd.cut(df['CreditScore'], bins=[350, 600, 700, 850], labels=['Poor', 'Fair', 'Good'])
    df = pd.get_dummies(df, columns=['CreditScore_Binned'], prefix='CreditScore')

    # Scale numerical features
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'Balance_to_Salary', 'Age_Tenure']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Handle class imbalance with SMOTE
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    return train_dataset, val_dataset, test_dataset, X_train.shape[1]