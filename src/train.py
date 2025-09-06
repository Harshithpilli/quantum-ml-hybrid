import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model_classical import ClassicalModel
from model_quantum import QuantumModel




def train_classical(data_path, model_out):
    df = pd.read_csv(data_path)
    X, y = df.drop("label", axis=1), df["label"]
    model = ClassicalModel()
    model.train(X, y)
    model.save(model_out)
    print(f"Classical model saved to {model_out}")




def train_quantum(data_path, model_out, epochs=50):
    df = pd.read_csv(data_path)
    X, y = df.drop("label", axis=1), df["label"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)[:, :4] # take first 4 features
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)


    model = QuantumModel(n_qubits=4, n_features=4, n_layers=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)


    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


    torch.save(model.state_dict(), model_out)
    print(f"Quantum model saved to {model_out}")




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/sample.csv")
    parser.add_argument("--model", type=str, choices=["classical", "quantum"], default="classical")
    parser.add_argument("--out", type=str, default="models/saved_model.pkl")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()


    if args.model == "classical":
        train_classical(args.data, args.out)
    else:
        train_quantum(args.data, args.out, epochs=args.epochs)