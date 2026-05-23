# IIoT Cybersecurity in Smart Manufacturing Systems

This project focuses on analyzing cybersecurity within Industrial Internet of Things (IIoT) smart manufacturing systems. It evaluates how different cybersecurity mechanisms impact cyber attacks, system downtime, and overall industrial efficiency using both synthetic data and real survey responses.

---

##  Project Overview

**Author:** Lea Mehmetaj  
**Academic Mentor:** Prof. Dr. Ylber Limani  

###  Objective
The main goal of this project is to evaluate the effectiveness of cybersecurity mechanisms in IIoT systems and their impact on:

- Reduction of cyber attacks (DoS, DDoS, Malware)
- Reduction of system downtime
- Improvement of manufacturing efficiency
- Comparison between synthetic dataset and survey results

---

##  Cybersecurity Mechanisms Analyzed

- Encryption Level
- Multi-Factor Authentication (MFA)
- IDS (Intrusion Detection System)
- AI-based Anomaly Detection
- Zero Trust Architecture

---

##  Datasets

### 1. Synthetic Dataset
- File: `Expanded_IIoT_Cybersecurity_Dataset.xlsx`
- 1000 samples
- 30 variables
- Includes attack metrics and system performance indicators

### 2. Survey Dataset
- File: `Pytesor_Rezultatet.xlsx`
- Responses from students and professionals
- Measures awareness and implementation of cybersecurity mechanisms

---

## Generated Visualizations

The Python script generates the following figures:

- 1_ROC_Curve_Krahasimi.png → ROC curves for all models
- 2_Krahasimi_Metrikave_BarChart.png → Accuracy & F1 comparison
- 3_mekanizmat_vs_sulme.png → Feature importance (XGBoost)
- 4_mekanizmat_vs_nderprerja.png → Regression on downtime
- 5_testimi_hipotezes.png → Advanced vs basic system comparison
- 6_lidhja_korrelacioni.png → Correlation heatmap
- 7_krahasimi_pytesor.png → AI vs survey comparison

---

## Technologies Used

- Python 3.11.9
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- OpenPyXL

---

##  How to Run

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

## How to Run the Project
1. Ensure you have the required libraries installed:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
    ```
2. Execute the main script:
   ```bash
   python main.py
     ```
