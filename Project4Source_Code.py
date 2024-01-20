import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image
import pandas as pd

# Step 1: Define the paths to your training and testing datasets
train_data_dir = 'extracted_dataset/DS_IDRID/Train'
test_data_dir = 'extracted_dataset/DS_IDRID/Test'

# Function to clean the label by removing '-'
def clean_label(label):
    return label.replace('-', '')

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if filename.endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)

        # Extract the label from the filename and clean it
        label = clean_label(image_path.split('_')[-1].split('.')[0])

        if label == '0':  # NonDR
            label = 0
        elif label in ['3', '4']:  # DR
            label = 1
        else:
            return None  # Discard images with Label 1 and Label 2

        if self.transform:
            image = self.transform(image)

        return image, label

# Step 3: Define data transformations and augmentations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Step 4: Create data loaders for training and testing
train_dataset = CustomDataset(data_dir=train_data_dir, transform=transform)
test_dataset = CustomDataset(data_dir=test_data_dir, transform=transform)

# Filter out None entries (images with Label 1 and Label 2)
train_dataset = [entry for entry in train_dataset if entry is not None]
test_dataset = [entry for entry in test_dataset if entry is not None]

# Step 5: Experiment with different hyperparameters (batch size, learning rate, max epochs)
experiments = [
    {'batch_size': 64, 'learning_rate': 0.001, 'num_epochs': 30},
    {'batch_size': 32, 'learning_rate': 0.01, 'num_epochs': 40},
    {'batch_size': 128, 'learning_rate': 0.0001, 'num_epochs': 20},
]

results = []  # Store results for different experiments

for experiment in experiments:
    batch_size = experiment['batch_size']
    learning_rate = experiment['learning_rate']
    num_epochs = experiment['num_epochs']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    y_true = []
    y_pred = []

    if len(test_loader.dataset) == 0:
        print("Testing dataset is empty.")
    else:
        model.eval()

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)

        if len(conf_matrix) == 2:
            true_positive = conf_matrix[1, 1]
            true_negative = conf_matrix[0, 0]
            false_positive = conf_matrix[0, 1]
            false_negative = conf_matrix[1, 0]

            sensitivity = true_positive / (true_positive + false_negative)
            specificity = true_negative / (true_negative + false_positive)
        else:
            sensitivity = 0.0
            specificity = 0.0

        # Display results for this experiment
        print("Experiment Results:")
        print(f"Batch Size: {batch_size}, Learning Rate: {learning_rate}, Num Epochs: {num_epochs}")
        print("Accuracy:", accuracy)
        print("Sensitivity (True Positive Rate):", sensitivity)
        print("Specificity (True Negative Rate):", specificity)
        print("Confusion Matrix:")
        print(conf_matrix)

        # Append results to the list
        results.append({
            'Batch Size': batch_size,
            'Learning Rate': learning_rate,
            'Num Epochs': num_epochs,
            'Accuracy': accuracy,
            'Sensitivity': sensitivity,
            'Specificity': specificity
        })

# Display results in a table
results_table = pd.DataFrame(results)
print("\nResults Table:")
print(results_table)