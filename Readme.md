# 🚔 Crime Trend Analysis & Forecasting Using Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge\&logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge\&logo=scikitlearn)
![Data Analytics](https://img.shields.io/badge/Data%20Analytics-Pandas-green?style=for-the-badge\&logo=pandas)
![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-red?style=for-the-badge\&logo=streamlit)
![Status](https://img.shields.io/badge/Project-Completed-success?style=for-the-badge)

### 📊 A Data-Driven Approach to Crime Trend Analysis and Forecasting Using Machine Learning Models

*Transforming Historical Crime Data into Predictive Intelligence*

</div>

---

# 📌 Project Overview

Crime has become a significant social challenge affecting public safety, economic development, and community well-being. Understanding crime patterns and forecasting future crime trends can help policymakers, law enforcement agencies, and researchers make informed decisions.

This project develops a **Machine Learning-based Crime Forecasting System** that analyzes historical crime records and predicts future violent crime rates using advanced data analytics and predictive modeling techniques.

The project integrates:

✅ Data Cleaning & Preprocessing
✅ Exploratory Data Analysis (EDA)
✅ Feature Engineering
✅ Machine Learning Modeling
✅ Crime Forecasting
✅ Model Evaluation
✅ Streamlit Deployment

The final model forecasts **next-year violent crime rates** based on historical crime patterns and regional crime characteristics.

---

# 🎯 Research Objectives

### Primary Objective

Develop a data-driven machine learning system capable of forecasting future crime trends using historical crime datasets.

### Specific Objectives

* Analyze historical crime patterns across counties
* Identify trends in violent, property, and firearm-related crimes
* Explore relationships among crime indicators
* Engineer predictive features from historical crime data
* Build and evaluate multiple machine learning models
* Forecast future violent crime rates
* Deploy an interactive web-based prediction system

---

# 🗂 Dataset Description

The dataset contains historical crime records collected from multiple counties over several years.

### Dataset Features

| Feature        | Description                           |
| -------------- | ------------------------------------- |
| County         | Administrative region                 |
| Year           | Crime reporting year                  |
| Violent Rate   | Violent crimes per population         |
| Property Rate  | Property crimes per population        |
| Firearm Rate   | Firearm-related crimes per population |
| Violent Count  | Number of violent crimes              |
| Property Count | Number of property crimes             |
| Firearm Count  | Number of firearm crimes              |

---

# 🔄 Project Workflow

```text
Data Collection
        ↓
Data Cleaning
        ↓
Missing Value Handling
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Machine Learning Modeling
        ↓
Model Evaluation
        ↓
Crime Forecasting
        ↓
Streamlit Deployment
```

---

# 🧹 Data Preprocessing

The dataset underwent extensive preprocessing to ensure data quality and model reliability.

### Preprocessing Tasks

* Removed duplicate records
* Checked missing values
* Interpolated firearm-related missing values
* Converted data types
* Sorted data by county and year
* Created county-wise historical sequences
* Prepared forecasting dataset

### Missing Value Treatment

Linear interpolation was used for firearm-related variables because crime data follows temporal trends and interpolation preserves continuity better than mean or median imputation.

```python
df[['Firearm Count','Firearm Rate']] = (
    df.groupby('County')
      [['Firearm Count','Firearm Rate']]
      .transform(
          lambda x: x.interpolate(method='linear')
      )
)
```

---

# 📊 Exploratory Data Analysis (EDA)

EDA was conducted to identify patterns, anomalies, and trends.

### Analyses Performed

### 📈 Crime Trends Over Time

* Violent crime evolution
* Property crime evolution
* Firearm crime evolution

### 🏙 County-Level Analysis

* High-crime counties
* Low-crime counties
* County comparisons

### 🔥 Correlation Analysis

* Relationships among crime indicators
* Correlation heatmaps

### 📉 Distribution Analysis

* Crime rate distributions
* Outlier identification

---

# ⚙️ Feature Engineering

Feature engineering was performed to improve forecasting performance.

## Lag Features

Historical crime rates from previous years:

```python
Violent_Lag1
Violent_Lag2
Property_Lag1
Property_Lag2
Firearm_Lag1
Firearm_Lag2
```

## Rolling Average Features

3-year moving averages:

```python
Violent_MA3
Property_MA3
Firearm_MA3
```

## Growth Features

Crime growth trends:

```python
Violent_Growth
Property_Growth
Firearm_Growth
```

## County Historical Features

Average county crime profiles:

```python
County_Avg_Violent
County_Avg_Property
County_Avg_Firearm
```

## Crime Risk Categories

Crime severity levels:

* Low Risk
* Medium Risk
* High Risk

---

# 🎯 Forecasting Target

Unlike traditional regression projects that predict current crime rates, this project predicts:

## Next-Year Violent Crime Rate

```python
df['Future_Violent_Rate'] =
df.groupby('County')['Violent Rate'].shift(-1)
```

This transforms the project into a genuine **Crime Forecasting System**.

---

# 🤖 Machine Learning Models

The following models were implemented and compared.

| Model                              | Purpose                 |
| ---------------------------------- | ----------------------- |
| Linear Regression                  | Baseline forecasting    |
| Random Forest Regressor            | Ensemble forecasting    |
| GridSearch Optimized Random Forest | Tuned forecasting model |

---

# 📏 Evaluation Metrics

Models were evaluated using:

### Mean Absolute Error (MAE)

Measures average prediction error.

### Root Mean Squared Error (RMSE)

Measures prediction accuracy.

### R² Score

Measures explanatory power of the model.

---

# 🔍 Model Validation

### Cross Validation

```python
cross_val_score()
```

Used to evaluate model stability and generalization.

### Hyperparameter Tuning

```python
GridSearchCV()
```

Used to optimize Random Forest performance.

---

# 📈 Forecasting Results

The optimized Random Forest model was able to accurately forecast future violent crime rates using historical crime patterns.

### Output

* Actual Crime Rate
* Predicted Crime Rate
* Forecast Accuracy
* Feature Importance Analysis

---

# 🏆 Feature Importance Analysis

Feature importance analysis was performed to identify the most influential factors affecting future crime rates.

Examples:

* Previous Violent Crime Rate
* Historical County Crime Average
* Property Crime Trends
* Firearm Crime Trends
* Crime Growth Rates

---

# 🌐 Streamlit Web Application

A user-friendly Streamlit application was developed for real-time forecasting.

### Features

✅ Interactive dashboard

✅ Crime forecasting interface

✅ Real-time predictions

✅ User-friendly visualization

### Run Application

```bash
streamlit run app.py
```

---

# 📂 Project Structure

```text
crime-trend-analysis/
│
├── data/
│   ├── Dataset.csv
│   └── cleaned_crime_dataset.csv
│
├── notebooks/
│   └── Crime_Analysis.ipynb
│
├── models/
│   ├── crime_prediction_model.pkl
│   └── scaler.pkl
│
├── visuals/
│
├── app.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/crime-trend-analysis.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Notebook

```bash
jupyter notebook
```

### Launch Streamlit

```bash
streamlit run app.py
```

---

# 📦 Requirements

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
joblib
```

---

# 🔮 Future Improvements

* ARIMA Time-Series Forecasting
* Facebook Prophet Forecasting
* LSTM Deep Learning Models
* Geo-Spatial Crime Mapping
* Interactive Crime Dashboards
* Real-Time Crime Data Integration
* Smart City Analytics Platform

---

# ⚖️ Ethical Considerations

Crime forecasting systems should be used responsibly.

Potential concerns include:

* Prediction bias
* Data fairness
* Privacy issues
* Misuse of predictive policing

The system is intended for educational, analytical, and research purposes only.

---

# 🎓 Academic Contribution

This project demonstrates expertise in:

* Data Analytics
* Predictive Analytics
* Machine Learning
* Data Visualization
* Feature Engineering
* Forecasting Systems
* Model Deployment

---

# 👨‍💻 Author

## Dilan Karunanayake

**Data Analytics & Machine Learning Enthusiast**

Final Year Undergraduate

---

<div align="center">

## ⭐ If you found this project useful, please give it a star ⭐

### 🚔 Turning Historical Crime Data into Predictive Intelligence 🚔

</div>
