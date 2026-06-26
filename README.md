## From Individual Calibration to Reliable Classifiers: ALD Parameterization with mPAIC Guarantees (ICML 2026)

## **Directory Structure**
```
datasets/
├── classification/
│   ├── adult/
│   │   └── adult.data                        # Adult Census Income dataset
│   ├── dry_bean/
│   │   └── dry_bean_dataset.csv              # Dry Bean dataset
│   ├── heart_disease/
│   │   └── heart_disease_uci.csv             # Heart Disease UCI dataset
│   ├── online_shoppers/
│   │   └── online_shoppers_intention.csv     # Online Shoppers Intention dataset
│   └── wdbc/
│       └── wdbc.data                         # Breast Cancer Wisconsin (Diagnostic) dataset
│
Figs/
├── *_run*.png                                # Training/Testing loss curves generated during execution
│
./
├── datasets.py                               # Data loading, preprocessing, and PyTorch Lightning DataModules
├── model_all.py                              # Neural network architectures (Softmax, ALD, ICALD, MMD variants)
├── script.py                                 # Main entry point for training, evaluation, and logging
├── utils.py                                  # Metric calculations (ECE, KCE, Group-ECE) and Kernel functions
├── results_summary.csv                       # Final aggregated metrics for all runs
│
requirements.txt                              # Python dependencies
README.md                                     # Project description and usage instructions
```
---

## **Requirements**
Install the required dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

---

## **Usage**

### **1. Data Preparation**
The code expects datasets to be placed in specific subdirectories within datasets/classification/. You must download the datasets (e.g., from UCI Machine Learning Repository) and place them as follows:

WDBC: datasets/classification/wdbc/wdbc.data

Adult: datasets/classification/adult/adult.data

Heart Disease: datasets/classification/heart_disease/heart_disease_uci.csv

Online Shoppers: datasets/classification/online_shoppers/online_shoppers_intention.csv

Dry Bean: datasets/classification/dry_bean/dry_bean_dataset.csv

### **2. Running Experiments**
Use `script.py` to train and evaluate models:
```bash
python script.py
```

