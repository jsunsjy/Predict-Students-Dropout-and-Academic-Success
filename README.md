# ReadMe File for Predicting Student Dropout and Graduation Using Enrollment Data
### A Learning Analytics Approach with Random Forests  
**Author:** Lisa Sun  
**Date:** 2025-10-01  

---

## 📌 Project Information
- **Title:** Predicting Student Dropout and Graduation Using Enrollment Data: A Learning Analytics Approach with Random Forests 
- **Principal Investigator:** Lisa Sun  
- **ORCID iD:** 0009-0000-6763-0355  
- **Affiliation:** Teachers College, Columbia University  
- **Email:** ls4116@tc.columbia.edu  
- **Language:** English  
- **Keywords:** Academic performance, Multi-class classification, Machine learning  

---

## 📖 Abstract
This project explores how higher education enrollment data can be leveraged to predict whether students are likely to graduate, remain enrolled, or drop out. Using the UCI “Predict Students’ Dropout and Academic Success” dataset (N = 4,424). Random Forest models were trained and evaluated to identify early indicators of student risk. The study emphasizes accuracy and fairness, assessing whether predictions vary across demographic and socioeconomic subgroups. The goal is to provide insights for universities developing early warning systems and equitable interventions to support student success.

---

## 📂 Data and File Overview

### **Filenames and Content**

- **`student_dropout_data.csv`**  
  Main dataset used for analysis.  
  - Contains **4,424 anonymized student records** with **36 attributes** (demographic, socioeconomic, macroeconomic, academic).  
  - Aggregated from **institutional student records**, the **CNAES admissions database**, and **PORDATA** (macroeconomic indicators).  
  - Each row = one student.  
  - Target variable = **final status**: *dropout*, *enrolled*, or *graduate*.  

- **`Code_pipeline.ipynb`**  
  Jupyter Notebook documenting the **end-to-end workflow**:  
  - Preprocessing & feature encoding  
  - Train/test splits  
  - Random Forest model training  
  - Evaluation metrics & subgroup fairness checks  
  - Generates graphs & tables used in results  

- **`Final_Results_Graphs.pdf`**  
  PDF summary of **key findings**:  
  - Feature importance rankings  
  - Confusion matrices  
  - Subgroup performance analyses  
  - Overall model metrics  
  *Designed for quick review without running code.*  

- **`README.md`**  
  The file you are reading.  
  - Documents **project context, dataset details, file descriptions, methods, and usage guidelines**.  
  - Serves as the **main reference** for understanding scope and purpose.  

### **Dataset Structure**

- Organized as a **self-contained data package**  
- All attributes stored in **CSV file**  
- Other files (code, results, docs) **reference the dataset directly**

### **Creation Date**

- **October 2022** → Dataset published on **Zenodo** (Realinho et al.)  
- **June 2025** → Dataset copy processed; Random Forest workflow created for Learning Analytics coursework  
- **October 2025** → README finalized with structured metadata & explanations  

### **Update History**

- **Oct 2022** → Dataset first published  
- **Jun 2025** → Dataset prepared, pipeline + outputs created  
- **Oct 2025** → Documentation finalized  

*Future updates may add additional ML models or fairness analyses.*  

### **Related Data**

- The CSV dataset is a **processed file** derived from raw institutional sources (not included here):  
  - **Academic Management System** (student records)  
  - **Support System for Teaching Activity**  
  - **CNAES** (National Competition for Access to Higher Education admissions data)  
  - **PORDATA** (national indicators: unemployment, inflation, GDP)  

---

## 🔑 Sharing and Access Information

### **📜 License**
- The dataset is distributed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.  
- This allows **reuse and adaptation** with proper credit to the original authors.  

### **📚 Publications Using This Data**
- Martins, M. V., Tolledo, D., Machado, J., Baptista, L. M. T., & Realinho, V. (2021).  
  *Early prediction of student’s performance in higher education: a case study.*  
  Springer. [https://doi.org/10.1007/978-3-030-72657-7_16](https://doi.org/10.1007/978-3-030-72657-7_16)  

### **🌐 Accessible Data Locations**
- **Zenodo (official archive with DOI):**  
  [https://doi.org/10.5281/zenodo.5777339](https://doi.org/10.5281/zenodo.5777339)  

- **UCI Machine Learning Repository (publicly accessible):**  
  [https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)  

### **📝 Recommended Citation**
> Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022).  
> *Predicting Student Dropout and Academic Success* \[Data set\]. Zenodo.  
> [https://doi.org/10.5281/zenodo.5777339](https://doi.org/10.5281/zenodo.5777339)  

---

## 🧪 Methodological Information

### 📊 Data Collection  
The original dataset, **Predict Students’ Dropout and Academic Success**, was created by aggregating data from institutional and national sources between **2008/2009 and 2018/2019**. Data were drawn from:  

- **Institutional records:** Academic Management System and Support System for Teaching Activity  
- **Admissions databases:** CNAES (National Competition for Access to Higher Education)  
- **Macroeconomic indicators:** PORDATA (unemployment, inflation, GDP)  

The dataset was formally published in *Data (MDPI)* (2022).  
Accessible at:  
- Zenodo: [https://doi.org/10.5281/zenodo.5777339](https://doi.org/10.5281/zenodo.5777339)  
- UCI Machine Learning Repository: [Dataset Link](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)  

**Reference:**  
> Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). *Predicting Student Dropout and Academic Success*. Data, 7(11), 146.  
> [https://doi.org/10.3390/data7110146](https://doi.org/10.3390/data7110146)  

### ⚙️ Data Processing  

Steps applied for this project:  
- Retrieved dataset from **UCI repository** using `ucimlrepo`  
- Excluded `1st_sem` and `2nd_sem` features → retained only **enrollment-time features**  
- Re-coded targets into binary classes:  
  - `0 = Dropout`  
  - `2 = Graduate`  
- Excluded students still enrolled to sharpen comparisons  
- Split into **80/20 training/testing sets** with stratification  
- Applied **oversampling** to balance classes (~50/50) in the training set  

**Preprocessing Syntax:**  

```python
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd

# Import dataset
ds = fetch_ucirepo(id=697)
X, y = ds.data.features, ds.data.targets

# Clean target labels
y = y.iloc[:,0].map({'Dropout':0, 'Enrolled':1, 'Graduate':2}).astype(int)

# Keep enrollment-only features
X = X.loc[:, ~X.columns.str.contains('(1st_sem|2nd_sem)', regex=True)]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Oversample minority (dropouts)
dropouts = X_train[y_train==0]
graduates = X_train[y_train==2]
dropouts_resampled = resample(dropouts, replace=True,
                              n_samples=len(graduates), random_state=42)
X_train_bal = pd.concat([dropouts_resampled, graduates])
```

### 💻 Software and Hardware
#### 🔧 Environment
- **Language:** Python 3.10  
- **Platform:** Google Colab (Linux backend)  

#### 📦 Libraries Used
- **pandas**, **numpy** → data handling and manipulation  
- **matplotlib**, **seaborn** → data visualization  
- **scikit-learn** → model training, evaluation, and metrics  
- **ucimlrepo** → direct dataset access  
- **Random Forest Classifier** → implemented using `sklearn.ensemble.RandomForestClassifier`  

#### 📜 Syntax

```python
# Data handling & visualization
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning model
from sklearn.ensemble import RandomForestClassifier

# Data splitting & evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
```

---

### ✅ Standards and Calibration
Variables were encoded consistently across student records (categorical values standardized, continuous values expressed in comparable numeric scales). No external calibration instruments were required.
