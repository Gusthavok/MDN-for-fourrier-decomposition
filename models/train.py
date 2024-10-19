import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


import torch.nn as nn
import torch.nn.functional as F

gamma = .1


class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        """
        input_size: Dimension of the input features
        hidden_layers: List where each element represents the number of neurons in that hidden layer
        output_size: Dimension of the output layer (number of classes or regression output)
        """
        super(MLP, self).__init__()
        
        layers = []
        
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
        
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers[:-1]:  
            x = F.leaky_relu(layer(x))       
        
        x = self.layers[-1](x)
        return x

def normalize(tensor):
    

def train(model, train_loader, criterion, optimizer, num_epochs):
    history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for inputs, x_value in train_loader:
            outputs = model(inputs)
            targets = (1-gamma) * outputs + gamma * dirach(x_value)
            targets_normalized = normalize(targets)
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        history.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    return history

def load_data_from_csv(csv_file):
    """
    Load data from a CSV file and return input features and target labels as tensors.
    
    Parameters:
    - csv_file: str, path to the CSV file
    
    Returns:
    - X: torch.Tensor, input features
    - y: torch.Tensor, target labels
    """

    df = pd.read_csv(csv_file)

    X = df.iloc[:, :-1].values  # Caractéristiques
    y = df.iloc[:, -1].values    # Cible

    # Convertir en tenseurs PyTorch
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Reshape pour la sortie

    return X_tensor, y_tensor

# Exemple d'utilisation
if __name__ == "__main__":
    # Charger les données depuis le fichier CSV
    csv_file = 'data.csv'  # Remplacez par le chemin vers votre fichier CSV
    X, y = load_data_from_csv(csv_file)

    # Créer un DataLoader
    train_dataset = data.TensorDataset(X, y)
    train_loader = data.DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Définir les paramètres du modèle
    input_size = X.shape[2]  # Nombre de caractéristiques
    hidden_layers = [32]  # Deux couches cachées avec 64 et 32 neurones respectivement
    output_size = 6           # Une sortie (pour la régression)

    # Instancier le modèle MLP
    model = MLP(input_size, hidden_layers, output_size)

    # Définir la fonction de perte et l'optimiseur
    criterion = nn.MSELoss()  # Erreur quadratique moyenne pour la régression
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimiseur Adam

    # Entraîner le modèle
    num_epochs = 20
    train(model, train_loader, criterion, optimizer, num_epochs)
