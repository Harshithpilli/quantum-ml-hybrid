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
│── dataset_generator.py   # Synthetic dataset creation
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

# Run evaluation
python evaluate.py 
