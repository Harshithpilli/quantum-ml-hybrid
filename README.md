# Quantum Malware Hunter 🔐⚡

Hybrid Quantum + Classical Machine Learning framework to detect unknown malware patterns in real-time.
This project demonstrates how quantum-enhanced ML can complement classical models for cybersecurity.

# Features

Synthetic dataset generation (clean, noisy, adversarial malware traffic).

Classical ML baseline (Random Forest).

Quantum ML model (Qiskit + variational circuits).

Stress-testing with noise, adversarial samples, and cross-validation.

Modular GitHub structure for easy extension.

# Project Structure

Quantum-malware-simple-arch/
│── data/                  # Generated datasets (CSV)

│── classical_model.py     # Classical ML training/evaluation

│── quantum_model.py       # Quantum ML training/evaluation

│── evaluate.py            # Unified evaluation & stress-tests

│── make_dataset.py   # Synthetic dataset creation

│── requirements.txt       # Python dependencies

│── README.md              # Project documentation

# Example Results

✅ Classical & Quantum models achieve high accuracy on clean data.
✅ Noisy & adversarial evaluation shows robustness under obfuscation attempts.

# Requirements

Python 3.9+

scikit-learn

pandas, numpy

qiskit

matplotlib (optional, for plotting)

# Installation

```bash
# Clone the repository
git clone https://github.com/Harshithpilli/quantum-ml-hybrid.git
cd quantum-ml-hybrid

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate    # Linux/Mac
# venv\Scripts\activate     # Windows

# Install requirements
pip install -r requirements.txt

# Generate dataset
python make_dataset.py

# Classical Model 
# For Clean
python src/train.py --data data/sample_clean.csv --model classical --out models/classical.pkl
python src/evaluate.py --data data/sample_clean.csv --model models/classical_clean.pkl --type classical

# For Adversarial
python src/train.py --data data/sample_adversarial.csv --model classical --out models/classical_adv.pkl
python src/evaluate.py --data data/sample_adversarial.csv --model models/classical_adv.pkl --type classical

# For Noisy
python src/train.py --data data/sample_noisy.csv --model classical --out models/classical_noisy.pkl
python src/evaluate.py --data data/sample_noisy.csv --model models/classical_noisy.pkl --type classical

# Quantum Model
# For Clean
python src/train.py --data data/sample_clean.csv --model quantum --out models/quantum_clean.pt --epochs 50
python src/evaluate.py --data data/sample_clean.csv --model models/quantum_clean.pt --type quantum

# For Noisy
python src/train.py --data data/sample_noisy.csv --model quantum --out models/quantum_noisy.pt --epochs 50
python src/evaluate.py --data data/sample_noisy.csv --model models/quantum_noisy.pt --type quantum

# For Adversarial
python src/train.py --data data/sample_adversarial.csv --model quantum --out models/quantum_adversarial.pt --epochs 50
python src/evaluate.py --data data/sample_adversarial.csv --model models/quantum_adversarial.pt --type quantum 
