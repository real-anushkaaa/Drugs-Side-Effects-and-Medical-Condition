![GGS2](https://github.com/user-attachments/assets/32a10826-deee-49b1-945b-8e92370ae6f4)

# 💊 Drugs, Side Effects and Medical Condition arrow_drop_up

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-red.svg)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![Internship](https://img.shields.io/badge/Internship-Unified%20Mentor-purple.svg)](https://unifiedmentor.com/)
[![License](https://img.shields.io/badge/License-Educational-green.svg)](#license)

## 🚀 Project Summary

Drugs, Side Effects and Medical Condition is a comprehensive data science platform that analyzes drug safety, efficacy, and patient experiences using real-world data from Drugs.com. Implementing a complete end-to-end machine learning workflow, from data collection to interactive dashboards, providing actionable insights for healthcare decision-making.

![image](https://github.com/user-attachments/assets/808c8d81-1ffc-42f1-aad7-d51902b51e82)
![image](https://github.com/user-attachments/assets/411ffbb5-8368-4f56-a805-4cf19ae4fe40)
![image](https://github.com/user-attachments/assets/3476663d-432c-4d8e-b779-8e9f9a4df01e)
![image](https://github.com/user-attachments/assets/7b6ebe11-7488-4fac-981f-1c184fadf57d)
![image](https://github.com/user-attachments/assets/fb5c53a5-3e5e-49a3-9dbf-03821550d826)
![image](https://github.com/user-attachments/assets/8139ad0f-6161-48d1-b0e0-9c41733254c5)



### ✨ Key Highlights
- 📊 **11MB+ Drug Dataset** - Comprehensive analysis of thousands of drug-condition pairs
- 🤖 **6 ML Models** - Classification, regression, and clustering models for drug analysis
- 📱 **3 Interactive Dashboards** - Streamlit applications for drug exploration and comparison
- 🔍 **Association Rule Mining** - Discover hidden patterns in drug-condition-side effect relationships
- 📈 **Complete Data Pipeline** - 7-step notebook workflow from data collection to deployment
- 💡 **Real-world Applications** - Safety analysis, risk assessment, and patient experience insights

## 🎓 About This Project

hands-on experience in:

- **End-to-End ML Pipeline Development** - From data collection to model deployment
- **Healthcare Domain Expertise** - Understanding drug safety and patient experience analysis
- **Interactive Dashboard Creation** - Building user-friendly data visualization tools
- **Professional Data Science Practices** - Code documentation, version control, and reproducible analysis
- **Business Problem Solving** - Translating healthcare challenges into data science solutions

## 🎯 Business Objectives

| Objective | Description | Key Features |
|-----------|-------------|--------------|
| **🛡️ Safety Analysis** | Identify safest and most effective drugs per medical condition | Side effect profiling, severity prediction |
| **⚠️ Risk Assessment** | Flag high-risk drugs with warnings | Alcohol, pregnancy, controlled substance alerts |
| **👥 Patient Experience** | Recommend drugs based on patient sentiment and ratings | NLP sentiment analysis, rating predictions |

## 🏗️ Project Architecture

```
Drugsmed/
├── 📊 data/                          # Datasets and processed files
│   ├── drugs_side_effects_drugs_com.csv   # Raw dataset (11MB)
│   ├── drugs_processed.csv               # Cleaned dataset (17MB)
│   ├── drugs_sample.csv                  # Sample dataset (6MB)
│   └── *.pkl                            # Metadata and feature artifacts
├── 📓 notebooks/                     # Complete analysis workflow
│   ├── 01_data_collection.ipynb          # Data loading and exploration
│   ├── 02_business_scenarios.ipynb       # Business case definition
│   ├── 03_data_cleaning_feature_engineering.ipynb
│   ├── 04_exploratory_data_analysis.ipynb
│   ├── 05_association_rule_mining.ipynb
│   ├── 06_machine_learning_models.ipynb
│   └── 07_interactive_dashboard.ipynb
├── 🤖 models/                        # Trained ML models
│   ├── best_classification_model_*.joblib
│   ├── best_regression_model_*.joblib
│   └── feature_*.joblib                   # Scalers and encoders
├── 🌐 src/                          # Interactive applications
│   ├── streamlit_app.py                  # Main dashboard
│   ├── streamlit_dashboard.py            # Alternative dashboard
│   └── enhanced_search_dashboard.py      # Advanced search interface
└── 🖼️ img/                          # Project screenshots and visuals
```

## 📋 Detailed Workflow

### 📖 Notebook Pipeline

| Step | Notebook | Description | Key Outputs |
|------|----------|-------------|-------------|
| 1️⃣ | **Data Collection** | Load and explore drug dataset | Dataset statistics, quality assessment |
| 2️⃣ | **Business Scenarios** | Define use cases and success metrics | KPIs, analytical framework |
| 3️⃣ | **Data Cleaning & Feature Engineering** | Data preprocessing and transformation | Clean dataset, engineered features |
| 4️⃣ | **Exploratory Data Analysis** | Comprehensive data visualization | Insights, patterns, correlations |
| 5️⃣ | **Association Rule Mining** | Apriori algorithm for pattern discovery | Drug-condition relationships |
| 6️⃣ | **Machine Learning Models** | Train and evaluate ML models | Trained models, performance metrics |
| 7️⃣ | **Interactive Dashboard** | Build Streamlit web application | Deployable dashboard |

### 🤖 Machine Learning Models

- **🎯 Classification Models**
  - Gradient Boosting Classifier (Side-effect severity prediction)
  - Logistic Regression (Drug safety classification)
- **📈 Regression Models**  
  - Linear Regression (Drug rating prediction)
- **🔍 Clustering Analysis**
  - Drug similarity grouping
- **📝 NLP Models**
  - Sentiment analysis (VADER, TextBlob)
  - Text preprocessing and feature extraction

### 🌐 Interactive Dashboards

1. **Main Dashboard** (`streamlit_app.py`) - Core drug analysis and comparison
2. **Standard Dashboard** (`streamlit_dashboard.py`) - Alternative interface
3. **Enhanced Search** (`enhanced_search_dashboard.py`) - Advanced search capabilities

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git (for cloning)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/drugsmed.git
cd drugsmed
```

2. **Create virtual environment** (recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### 🏃‍♂️ Running the Project

#### Option 1: Interactive Dashboard (Recommended)
```bash
streamlit run src/streamlit_app.py
```
Then open your browser to `http://localhost:8501`

#### Option 2: Full Analysis Pipeline
```bash
jupyter lab
```
Run notebooks in sequence: `01` → `02` → `03` → `04` → `05` → `06` → `07`

#### Option 3: Alternative Dashboards
```bash
# Standard dashboard
streamlit run src/streamlit_dashboard.py

# Enhanced search interface
streamlit run src/enhanced_search_dashboard.py
```

## 📊 Dataset Information

| Attribute | Details |
|-----------|---------|
| **Source** | Drugs.com |
| **Size** | ~11MB (raw), ~17MB (processed) |
| **Records** | Thousands of drug-condition pairs |
| **Features** | 17+ columns including drug names, side effects, ratings, reviews, warnings |
| **Coverage** | Multiple therapeutic areas and drug classes |

### Key Data Fields
- 💊 **Drug Information**: Names, generic/brand classifications
- 🏥 **Medical Conditions**: Primary and secondary indications  
- ⭐ **Patient Ratings**: Effectiveness and satisfaction scores
- 📝 **Reviews**: Patient experiences and testimonials
- ⚠️ **Warnings**: Pregnancy, alcohol, controlled substance alerts
- 🔍 **Side Effects**: Frequency and severity classifications

## 🛠️ Technology Stack

### 📈 Data Science & ML
- **Core**: Pandas, NumPy, SciPy
- **Machine Learning**: Scikit-learn, Imbalanced-learn
- **NLP**: NLTK, TextBlob, VADER Sentiment
- **Association Mining**: Apyori, MLxtend

### 📊 Visualization
- **Static**: Matplotlib, Seaborn
- **Interactive**: Plotly, WordCloud
- **Web**: Streamlit, Streamlit-Plotly

### 🔧 Development
- **Environment**: Jupyter Lab/Notebook
- **Data Processing**: OpenPyXL, TQDM

## 💡 Use Cases & Applications

### 🏥 Healthcare Professionals
- Compare drug safety profiles
- Identify high-risk medications
- Analyze patient satisfaction trends
- Evidence-based treatment selection

### 🔬 Researchers
- Drug effectiveness analysis
- Side effect pattern discovery
- Patient experience research
- Pharmaceutical market analysis

### 📊 Data Scientists
- Complete ML pipeline example
- Text analytics implementation
- Interactive dashboard development
- Association rule mining techniques

## 📈 Key Results & Insights

- ✅ **Trained 6 ML models** with validated performance metrics
- 🔍 **Discovered drug-condition associations** using market basket analysis
- 📊 **Built 3 interactive dashboards** for different user needs
- 🎯 **Achieved accurate predictions** for drug safety and effectiveness
- 📝 **Processed thousands of patient reviews** with sentiment analysis
---

*Dedicated to the healthcare and data science community* 