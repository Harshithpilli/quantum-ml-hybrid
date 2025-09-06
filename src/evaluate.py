# ================================
# File: src/evaluate.py (Enhanced with Stress Tests)
# ================================
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from model_classical import ClassicalModel
from model_quantum import QuantumModel


def adversarial_samples(X, epsilon=0.1):
    perturb = np.sign(np.random.randn(*X.shape)) * epsilon
    return X + perturb


# ================= Classical Evaluation =================
def evaluate_classical(data_path, model_path):
    df = pd.read_csv(data_path)
    X, y = df.drop("label", axis=1), df["label"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Load trained model
    model = ClassicalModel()
    model.load(model_path)

    # Evaluate on test set
    preds = model.predict(X_test)
    print("\n[Classical Model - Test Set Evaluation]")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Accuracy:", accuracy_score(y_test, preds))

    # Cross-validation
    clf = RandomForestClassifier()
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print("Cross-Validation Accuracy:", cv_scores.mean())

    # Adversarial samples
    X_adv = adversarial_samples(X_test.to_numpy())
    preds_adv = model.predict(X_adv)
    print("\n[Classical Model - Adversarial Evaluation]")
    print(classification_report(y_test, preds_adv))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds_adv))
    print("Accuracy:", accuracy_score(y_test, preds_adv))


# ================= Quantum Evaluation =================
def evaluate_quantum(data_path, model_path):
    df = pd.read_csv(data_path)
    X, y = df.drop("label", axis=1), df["label"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)[:, :4]
    X_test_scaled = scaler.transform(X_test)[:, :4]

    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Load trained quantum model
    model = QuantumModel(n_qubits=4, n_features=4, n_layers=3)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        outputs = model(X_test_tensor)
        preds = torch.argmax(outputs, dim=1)

    print("\n[Quantum Model - Test Set Evaluation]")
    print(classification_report(y_test_tensor, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test_tensor, preds))
    print("Accuracy:", accuracy_score(y_test_tensor, preds))

    # Adversarial samples
    X_adv = adversarial_samples(X_test_scaled)
    X_adv_tensor = torch.tensor(X_adv, dtype=torch.float32)

    with torch.no_grad():
        outputs_adv = model(X_adv_tensor)
        preds_adv = torch.argmax(outputs_adv, dim=1)

    print("\n[Quantum Model - Adversarial Evaluation]")
    print(classification_report(y_test_tensor, preds_adv))
    print("Confusion Matrix:\n", confusion_matrix(y_test_tensor, preds_adv))
    print("Accuracy:", accuracy_score(y_test_tensor, preds_adv))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/sample.csv")
    parser.add_argument("--model", type=str, default="models/saved_model.pkl")
    parser.add_argument("--type", type=str, choices=["classical", "quantum"], default="classical")
    args = parser.parse_args()

    if args.type == "classical":
        evaluate_classical(args.data, args.model)
    else:
        evaluate_quantum(args.data, args.model)
