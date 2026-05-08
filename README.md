# 🚨 Police Stop Outcome Predictor

A machine learning web application that predicts the outcome of police traffic stops — whether an arrest is likely or not — based on driver demographics and stop details. Built with **Python**, **Scikit-learn**, and **Streamlit** with a professional navy blue dashboard UI.

---

## 📸 Preview

> A secure login dashboard leads into a two-panel prediction interface with real-time ML inference, probability bars, and CSV export — all styled in a dark navy blue theme.

---

## 🧠 How It Works

```
Police Data.csv
      ↓
Data Cleaning & EDA (Police_File.ipynb)
      ↓
Logistic Regression Model Training
      ↓
police_arrest_model.pkl (saved model)
      ↓
Streamlit App (App.py) loads model
      ↓
User inputs stop details → Prediction + Probability
```

---

## 📁 Project Structure

```
police-stop-predictor/
│
├── App.py                      # Streamlit web application
├── Police_File.ipynb           # Data analysis, EDA & model training notebook
├── police_arrest_model.pkl     # Trained Logistic Regression model
├── columns.pkl                 # Saved feature column names
├── Police Data.csv             # Raw dataset
└── README.md                   # Project documentation
```

---

## ⚙️ Features

- 🔐 **Secure Login** — Session-based authentication with username/password
- 📊 **EDA & Visualizations** — Gender, race, age distributions + correlation heatmap
- 🤖 **ML Model** — Logistic Regression trained on real police stop data
- 📈 **Probability Scores** — Shows arrest vs no-arrest confidence scores
- 📥 **CSV Export** — Download prediction results as CSV
- 🎨 **Premium UI** — Navy blue glassmorphism dark theme with custom fonts
- 📋 **Session History** — Tracks last prediction in session

---

## 🧪 Model Details

| Property | Value |
|---|---|
| Algorithm | Logistic Regression |
| Library | Scikit-learn |
| Train/Test Split | 70% / 30% |
| Max Iterations | 10,000 |
| Evaluation | Accuracy Score, Confusion Matrix, ROC-AUC Curve |

### Features Used for Prediction
| Feature | Description |
|---|---|
| `driver_age` | Age of the driver |
| `driver_gender` | Male / Female (one-hot encoded) |
| `driver_race` | White / Black / Hispanic / Asian / Other |
| `search_conducted` | Was a search performed? (0/1) |
| `drugs_related_stop` | Was it drug-related? (0/1) |

### Target Variable
- `is_arrested` → `1` (Arrested) or `0` (Not Arrested)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Hammad-Ahmed-hk/police-stop-predictor.git
cd police-stop-predictor
```

### 2. Create Virtual Environment (Recommended)
```bash
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install Dependencies
```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
```

### 4. Run the Notebook (to train model)
Open `Police_File.ipynb` in Jupyter and run all cells. This will generate:
- `police_arrest_model.pkl`
- `columns.pkl`

### 5. Run the App
```bash
streamlit run App.py
```

### 6. Login Credentials (Demo)
| Username | Password |
|---|---|
| `admin` | `password123` |
| `user` | `userpass` |

---

## 📦 Requirements

```
python >= 3.11
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
pickle (built-in)
```

---

## 📊 EDA Highlights (from Notebook)

The notebook performs the following analysis on `Police Data.csv`:

- **Driver Gender Distribution** — Bar chart of male vs female stops
- **Driver Race Distribution** — Count plot across racial groups
- **Driver Age Distribution** — Histogram with KDE curve
- **Gender vs Age** — Box plot comparison
- **Correlation Heatmap** — Numeric feature relationships
- **Confusion Matrix** — Model correct vs incorrect predictions
- **ROC Curve** — Model performance with AUC score

---

## 🌐 Deployment

This app can be deployed on **Streamlit Community Cloud** for free:

1. Push your code to GitHub
2. Go to 🔗 [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `App.py` as the main file
5. Click **Deploy**

> ⚠️ Make sure `police_arrest_model.pkl` and `columns.pkl` are included in your repo before deploying.

---

## ⚠️ Disclaimer

> This project is built for **educational and demonstration purposes only**. The predictions made by this model are probabilistic and should **never** be used as the basis for real law enforcement decisions. The model reflects patterns in historical data and may contain inherent biases.

---

## 👨‍💻 Author

**Hammad Ahmed**
- GitHub: [@Hammad-Ahmed-hk](https://github.com/Hammad-Ahmed-hk)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">
  Made with ❤️ using Python & Streamlit
</div>
