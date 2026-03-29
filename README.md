# 🏠 House Price Prediction Dashboard

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/ML-Random_Forest-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458?logo=pandas)

*End-to-end ML application for batch house price prediction with interactive analytics*

[Portfolio](https://decodedbyfarhan.tech) • [LinkedIn](https://www.linkedin.com/in/farhan16/)

</div>

---

## 📌 Overview

A production-ready machine learning application that enables users to upload datasets, generate house price predictions at scale, and analyze results through an interactive Streamlit dashboard.

### 🎯 What It Does
- Upload CSV datasets for batch prediction
- Automatic validation and preprocessing
- ML-powered price predictions using Random Forest
- Interactive analytics dashboard with KPIs
- Export results as CSV

---

## ✨ Key Features

- 📁 **CSV Upload** - Batch prediction interface
- 🔍 **Auto Validation** - Column checking and error handling
- 🤖 **ML Pipeline** - Random Forest with preprocessing
- 📊 **Dashboard** - Interactive charts and KPIs
- 📥 **Export** - Download predictions as CSV
- 🎨 **Clean UI** - User-friendly Streamlit interface

---

## 🛠️ Tech Stack

- **Python** - Core programming language
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning
- **Streamlit** - Web dashboard
- **Joblib** - Model serialization

---

## 📊 Model Details

**Algorithm:** Random Forest Regressor

**Preprocessing Pipeline:**
- Missing value imputation
- Feature scaling (StandardScaler)
- One-hot encoding for categorical features

**Evaluation Metrics:**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 📂 Project Structure

```
housing-ml-app/
│
├── app.py                  # Streamlit dashboard
├── train_model.py          # Model training script
├── requirements.txt        # Dependencies
│
├── data/
│   └── housing.csv         # Training dataset
│
├── models/
│   ├── model.pkl           # Trained model
│   └── pipeline.pkl        # Preprocessing pipeline
│
├── input/
│   └── sample_input.csv    # Sample test data
│
└── output/
    └── predictions.csv     # Generated predictions
```

---

## 📂 Dataset

**California Housing Dataset** - Built-in scikit-learn dataset with ~20,000 property records.

**Features:**
- `median_income` - Median household income
- `total_rooms` - Total rooms in block
- `population` - Block population
- `households` - Number of households
- `ocean_proximity` - Location category (categorical)
- Additional geographic and housing features

**Target:** `median_house_value` (house price in dollars)

---

## ⚙️ Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/farhann-16/house-price-prediction.git
cd house-price-prediction
```

### 2. Create Environment (Optional)

```bash
conda create -n housepriceenv python=3.10
conda activate housepriceenv
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train Model

```bash
python train_model.py
```

### 5. Run Application

```bash
streamlit run app.py
```

App runs at: `http://localhost:8501`

---

## 🧪 How to Use

1. **Launch** the Streamlit app
2. **Upload** a CSV file (or use `sample_input.csv`)
3. **Validate** - System checks required columns
4. **Predict** - Click "Run Prediction" button
5. **Analyze** - View results in dashboard
6. **Download** - Export predictions as CSV

---

## 🎯 Key Highlights

- ✅ End-to-end ML pipeline (training → deployment)
- ✅ Batch prediction system for real-world use
- ✅ Automated data validation and preprocessing
- ✅ Interactive analytics dashboard
- ✅ Production-ready code with error handling

---

## 🚀 Deployment

### Streamlit Cloud

1. Push to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect repository and deploy

### Docker

```bash
docker build -t house-price-app .
docker run -p 8501:8501 house-price-app
```

---

## 🤝 Contributing

Contributions welcome! Fork the repository and submit a pull request.

---

## 📝 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Farhan Diwan**

<div align="center">

[![Portfolio](https://img.shields.io/badge/Portfolio-decodedbyfarhan.tech-blue?style=for-the-badge)](https://decodedbyfarhan.tech)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-farhan16-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/farhan16/)
[![GitHub](https://img.shields.io/badge/GitHub-farhann--16-181717?style=for-the-badge&logo=github)](https://github.com/farhann-16)

</div>

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

**Made with ❤️ by Farhan**

</div>
