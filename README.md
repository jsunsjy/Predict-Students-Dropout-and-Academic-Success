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
  The current file.  
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
- **Jun 2025** → Dataset prepared, pipeline, and outputs created  
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

### 📄 Standards and Calibration
Variables were encoded consistently across student records (categorical values standardized, continuous values expressed in comparable numeric scales). No external calibration instruments were required.

### 🌡️ Quality Assurance
Quality checks ensured the dataset and modeling pipeline were reliable and reproducible. The steps below document missing-value screening, categorical code validation, and class-balance treatment (train-only).

#### Checks Performed
- No Missing Values — Validated that all features contained non-missing values.
- Consistent Categorical Coding — Verified that key categorical fields use only valid, expected codes.
- Balanced Training Set — To reduce bias toward the majority outcome, the binary training set was balanced via random oversampling of the minority class. Oversampling was applied only to the training split to prevent leakage into the test set.

```python
# Check for missing values across all features
missing_total = int(X.isna().sum().sum())
print(f"Missing values in X (should be 0): {missing_total}")

# Inspect unique codes for selected categorical variables
cat_cols = [
    "marital_status",
    "nationality",
    "mother_qualification",
    "father_qualification",
    "mother_occupation",
    "father_occupation",
    "application_mode",
    "daytime_evening_attendance",
    "gender",
    "scholarship_holder",
    "displaced",
    "educational_special_needs",
    "tuition_fees_up_to_date",
    "international_student"
]

# Only check columns that exist in X to avoid KeyError
for col in [c for c in cat_cols if c in X.columns]:
    print(f"{col}: {sorted(X[col].unique().tolist())[:25]}{' ...' if X[col].nunique() > 25 else ''}")

from sklearn.utils import resample
import pandas as pd

# Context: assumes X_train_binary and y_train_binary were created earlier
#          (e.g., by filtering the original 3-class target to {0=Dropout, 2=Graduate})

df_train = X_train_binary.copy()
df_train["Target"] = y_train_binary

# Split into class-specific frames
dropouts  = df_train[df_train["Target"] == 0]
graduates = df_train[df_train["Target"] == 2]

# Oversample the minority class to match the majority size
if len(dropouts) < len(graduates):
    dropouts_bal = resample(dropouts, replace=True,
                            n_samples=len(graduates), random_state=42)
    df_balanced = pd.concat([dropouts_bal, graduates], axis=0)
else:
    graduates_bal = resample(graduates, replace=True,
                             n_samples=len(dropouts), random_state=42)
    df_balanced = pd.concat([dropouts, graduates_bal], axis=0)

# Shuffle the balanced training set
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Confirm balance
class_pct = (df_balanced["Target"].value_counts(normalize=True) * 100).round(1)
print("Balanced train class %:\n", class_pct)

# Final X/y for training
X_train_bal = df_balanced.drop(columns="Target")
y_train_bal = df_balanced["Target"]
```

#### Quick Summary
- Missing values: None detected in feature matrix.
- Categorical codes: Spot-checked across demographic and socioeconomic fields for validity.
- Class balance (train set): Achieved ~50/50 distribution between Dropout (0) and Graduate (2) via oversampling; test distribution left untouched.

### Codes & Outliers

**Coding conventions**

- No placeholder codes are used; the released dataset is fully observed per the source documentation.  
- Target labels are integer-encoded: `0 = Dropout`, `1 = Enrolled`, `2 = Graduate`.  
- Features are either numeric (continuous) or integer-coded categorical, as listed in the data dictionary.

**Outliers**

- Potential outliers were addressed during dataset preparation by the original authors, as described in the published documentation.  
- This project retains enrollment-time variables without additional winsorization or trimming.

### Contributors
- Dataset authors: Valentim Realinho, Jorge Machado, Luís Baptista, and Mónica V. Martins.
- Current analysis & modeling: Lisa Sun (Teachers College, Columbia University) — Random Forest modeling focused on enrollment-only features to predict dropout versus graduation.

---

## Data-Specific Information

### 1) Number of Variables and Cases
- **Cases (rows):** 4,424 student records  
- **Variables (columns):** 36 attributes

> **Reproducible counts (optional)**
>
> ```python
> import pandas as pd
> from ucimlrepo import fetch_ucirepo
> ds = fetch_ucirepo(id=697)
> X, y = ds.data.features, ds.data.targets
> print({"rows": X.shape[0], "cols": X.shape[1]})
> ```

### 2) Variable List — Data Dictionary (36 Variables)

<details>
<summary><b>Expand Data Dictionary</b></summary>

<table>
  <thead>
    <tr>
      <th>Variable Name (Raw)</th>
      <th>Full Name</th>
      <th>Definition / Description</th>
      <th>Data Type</th>
      <th>Units of Measurement</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>marital_status</td>
      <td>Marital Status</td>
      <td>Student’s marital status at enrollment (1 = Single, 2 = Married, 3 = Widower, 4 = Divorced, etc.).</td>
      <td>Integer (Categorical)</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td>application_mode</td>
      <td>Application Mode</td>
      <td>Admission pathway (general contingent, transfer, international, ordinance-based, etc.).</td>
      <td>Integer (Categorical)</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td>application_order</td>
      <td>Application Order</td>
      <td>Priority of the program chosen (0 = First choice; 9 = Last choice).</td>
      <td>Integer</td>
      <td>Rank (0–9)</td>
    </tr>
    <tr>
      <td>course</td>
      <td>Course</td>
      <td>Degree program (e.g., Nursing, Informatics Engineering, Management).</td>
      <td>Integer (Categorical)</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td>daytime_evening_attendance</td>
      <td>Daytime/Evening Attendance</td>
      <td>Attendance mode (1 = Daytime; 0 = Evening).</td>
      <td>Integer (Binary)</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td>previous_qualification</td>
      <td>Previous Qualification</td>
      <td>Highest education level before admission (Secondary, Bachelor’s, Master’s, Doctorate).</td>
      <td>Integer (Categorical)</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td>previous_qualification_grade</td>
      <td>Previous Qualification Grade</td>
      <td>Grade from prior qualification on a 0–200 scale.</td>
      <td>Continuous (Numeric)</td>
      <td>Score (0–200)</td>
    </tr>
    <tr>
      <td>nationality</td>
      <td>Nationality</td>
      <td>Student’s nationality (integer-coded; e.g., 1 = Portuguese, 41 = Brazilian, 109 = Colombian).</td>
      <td>Integer (Categorical)</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td>mother_qualification</td>
      <td>Mother’s Qualification</td>
      <td>Mother’s highest education level (Secondary, Higher Education, Master’s, Doctorate).</td>
      <td>Integer (Categorical)</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td>father_qualification</td>
      <td>Father’s Qualification</td>
      <td>Father’s highest education level (same coding as above).</td>
      <td>Integer (Categorical)</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td>mother_occupation</td>
      <td>Mother’s Occupation</td>
      <td>Mother’s occupation group (e.g., health professional, teacher, administrative staff, unskilled worker).</td>
      <td>Integer (Categorical)</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td>father_occupation</td>
      <td>Father’s Occupation</td>
      <td>Father’s occupation group (coded similarly to mother’s).</td>
      <td>Integer (Categorical)</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td>admission_grade</td>
      <td>Admission Grade</td>
      <td>Final admission grade (0–200 scale).</td>
      <td>Continuous (Numeric)</td>
      <td>Score (0–200)</td>
    </tr>
    <tr>
      <td>displaced</td>
      <td>Displaced Student</td>
      <td>Indicates if the student relocated to attend (1 = Yes; 0 = No).</td>
      <td>Integer (Binary)</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td>educational_special_needs</td>
      <td>Educational Special Needs</td>
      <td>Registered special needs (1 = Yes; 0 = No).</td>
      <td>Integer (Binary)</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td>debtor</td>
      <td>Debtor</td>
      <td>Presence of unpaid tuition fees (1 = Yes; 0 = No).</td>
      <td>Integer (Binary)</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td>tuition_fees_up_to_date</td>
      <td>Tuition Fees Up to Date</td>
      <td>Tuition payments fully up to date (1 = Yes; 0 = No).</td>
      <td>Integer (Binary)</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td>gender</td>
      <td>Gender</td>
      <td>Student’s gender (1 = Male; 0 = Female).</td>
      <td>Integer (Binary)</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td>scholarship_holder</td>
      <td>Scholarship Holder</td>
      <td>Indicates scholarship status (1 = Yes; 0 = No).</td>
      <td>Integer (Binary)</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td>age_at_enrollment</td>
      <td>Age at Enrollment</td>
      <td>Student age at the time of enrollment.</td>
      <td>Integer</td>
      <td>Years</td>
    </tr>
    <tr>
      <td>international_student</td>
      <td>International Student</td>
      <td>Enrollment as an international student (1 = Yes; 0 = No).</td>
      <td>Integer (Binary)</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td>cu_1st_sem_credited</td>
      <td>Curricular Units 1st Sem (Credited)</td>
      <td>Number of curricular units credited in the first semester.</td>
      <td>Integer</td>
      <td>Count</td>
    </tr>
    <tr>
      <td>cu_1st_sem_enrolled</td>
      <td>Curricular Units 1st Sem (Enrolled)</td>
      <td>Number of curricular units enrolled in during the first semester.</td>
      <td>Integer</td>
      <td>Count</td>
    </tr>
    <tr>
      <td>cu_1st_sem_evaluations</td>
      <td>Curricular Units 1st Sem (Evaluations)</td>
      <td>Number of evaluations/tests taken in the first semester.</td>
      <td>Integer</td>
      <td>Count</td>
    </tr>
    <tr>
      <td>cu_1st_sem_approved</td>
      <td>Curricular Units 1st Sem (Approved)</td>
      <td>Number of curricular units passed in the first semester.</td>
      <td>Integer</td>
      <td>Count</td>
    </tr>
    <tr>
      <td>cu_1st_sem_grade</td>
      <td>Curricular Units 1st Sem (Grade)</td>
      <td>Average grade in the first semester.</td>
      <td>Continuous (Numeric)</td>
      <td>Score (0–20)</td>
    </tr>
    <tr>
      <td>cu_1st_sem_without_evaluations</td>
      <td>Curricular Units 1st Sem (No Evaluation)</td>
      <td>Number of curricular units without evaluations in the first semester.</td>
      <td>Integer</td>
      <td>Count</td>
    </tr>
    <tr>
      <td>cu_2nd_sem_credited</td>
      <td>Curricular Units 2nd Sem (Credited)</td>
      <td>Number of curricular units credited in the second semester.</td>
      <td>Integer</td>
      <td>Count</td>
    </tr>
    <tr>
      <td>cu_2nd_sem_enrolled</td>
      <td>Curricular Units 2nd Sem (Enrolled)</td>
      <td>Number of curricular units enrolled in during the second semester.</td>
      <td>Integer</td>
      <td>Count</td>
    </tr>
    <tr>
      <td>cu_2nd_sem_evaluations</td>
      <td>Curricular Units 2nd Sem (Evaluations)</td>
      <td>Number of evaluations/tests taken in the second semester.</td>
      <td>Integer</td>
      <td>Count</td>
    </tr>
    <tr>
      <td>cu_2nd_sem_approved</td>
      <td>Curricular Units 2nd Sem (Approved)</td>
      <td>Number of curricular units passed in the second semester.</td>
      <td>Integer</td>
      <td>Count</td>
    </tr>
    <tr>
      <td>cu_2nd_sem_grade</td>
      <td>Curricular Units 2nd Sem (Grade)</td>
      <td>Average grade in the second semester.</td>
      <td>Continuous (Numeric)</td>
      <td>Score (0–20)</td>
    </tr>
    <tr>
      <td>cu_2nd_sem_without_evaluations</td>
      <td>Curricular Units 2nd Sem (No Evaluation)</td>
      <td>Number of curricular units without evaluations in the second semester.</td>
      <td>Integer</td>
      <td>Count</td>
    </tr>
    <tr>
      <td>unemployment_rate</td>
      <td>Unemployment Rate</td>
      <td>National unemployment rate during the enrollment year.</td>
      <td>Continuous (Numeric)</td>
      <td>Percent (%)</td>
    </tr>
    <tr>
      <td>inflation_rate</td>
      <td>Inflation Rate</td>
      <td>National inflation rate during the enrollment year.</td>
      <td>Continuous (Numeric)</td>
      <td>Percent (%)</td>
    </tr>
    <tr>
      <td>gdp</td>
      <td>Gross Domestic Product (GDP)</td>
      <td>National GDP indicator for the enrollment year.</td>
      <td>Continuous (Numeric)</td>
      <td>Currency (Index)</td>
    </tr>
    <tr>
      <td>target</td>
      <td>Outcome (Target Variable)</td>
      <td>Final student outcome: 0 = Dropout, 1 = Enrolled, 2 = Graduate.</td>
      <td>Categorical (3-class)</td>
      <td>N/A</td>
    </tr>
  </tbody>
</table>

</details>

### 3) Units of Measurement
- **Grades (admission / previous qualification):** 0–200 scale  
- **Semester grades:** 0–20 scale  
- **Age:** Years  
- **Macroeconomic indicators:** Percent (%) for unemployment and inflation; GDP as indexed currency  
- **Counts:** Integer counts for curricular-unit fields (credited, enrolled, approved, evaluations)

### 4) Missing-Data Codes
- No missing-value codes are present

### 5) Specialized Formats / Abbreviations
- Distributed as **CSV (UTF-8)**  
- Categorical variables stored as **integer codes**  
- Outcome variable **target** encoded as `0 = Dropout`, `1 = Enrolled`, `2 = Graduate

---

## Reflections

### 1. Which metadata standard did you choose and why?
I chose the Data Documentation Initiative (DDI) standard because it is widely recognized in research and keeps metadata consistent, structured, and reusable. DDI provides clear guidance for documenting variables, methods, and provenance, which improves human readability and machine-actionability. Aligning with DDI raised the overall professionalism of this README and supports long-term reproducibility.

### 2. Which template/software did you use?
I used the GitHub repository editor. This choice minimized setup time and leveraged tools already in daily use, which reduced friction and kept attention on content quality. Working directly in GitHub ensured every change was versioned, reviewable, and diffable, and the built-in Markdown preview helped validate layout continuously.

### 3. What was the most challenging part of creating a README file? How did you overcome these obstacles?
The most challenging part was translating code and dataset details into concise documentation that remains accurate and accessible. To overcome this, I decomposed the analysis into small, goal-oriented steps, wrote plain-language explanations before adding code, and aligned the variable documentation to DDI conventions. Iterative previews in GitHub ensured the formatting rendered correctly, and a consistent style guide (headings, tables, and code blocks) kept the document coherent and professional.
