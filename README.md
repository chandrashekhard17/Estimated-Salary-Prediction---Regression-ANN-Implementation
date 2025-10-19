# 💰 Customer Estimated Salary Prediction Using ANN (Regression)

## 📘 Project Overview
This project predicts the **estimated salary of a customer** using an **Artificial Neural Network (ANN)** regression model.  
The model is trained on the **Churn Modelling dataset** which contains customer details such as Credit Score, Geography, Gender, Age, Tenure, Balance, Number of Products, and more.

The goal of this project is to demonstrate how **ANNs can be used for regression tasks** to predict continuous numeric values, such as salaries.

---

## 🚀 Features
- Predicts **estimated salary** for a customer using ANN regression  
- Interactive **Streamlit web app** for easy predictions  
- Preprocessing includes **Label Encoding**, **One-Hot Encoding**, and **feature scaling**  
- Modular and clean code, ready for deployment

---

## 🧩 Tech Stack
- **Python** – Programming Language  
- **TensorFlow / Keras** – ANN model building  
- **Pandas & NumPy** – Data processing  
- **Scikit-learn** – Preprocessing and scaling  
- **Streamlit** – Web interface  
- **Pickle** – Saving encoders and scalers  

---

## 🧱 Project Structure
```
Customer-Salary-Prediction/
│
├── app.py                 # Streamlit app for salary prediction
├── model.h5               # Pretrained ANN regression model
├── scaler.pkl             # Scaler for input features
├── label_encoder_gender.pkl # Encoder for Gender column
├── onehot_encoder_geo.pkl   # Encoder for Geography column
├── data/                  # Churn Modelling dataset CSV
└── README.md              # Project documentation
```

---

## ⚙️ Installation and Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/chandrashekhard17/Estimated-Salary-Prediction---Regression-ANN-Implementation.git
```

### 2️⃣ Navigate to the project directory
```bash
cd Customer-Salary-Prediction
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit app
```bash
streamlit run app.py
```

---

## 🧠 ANN Regression Explained (Deep Learning)

**Artificial Neural Networks (ANNs)** are a type of deep learning model inspired by the human brain. They consist of **layers of interconnected neurons**:

1. **Input Layer:** Receives the features of the dataset (Credit Score, Age, Balance, etc.)  
2. **Hidden Layers:** Apply **weights, biases, and activation functions** to capture complex patterns  
3. **Output Layer:** For regression, the **output neuron predicts a continuous numeric value**  

### Key Concepts for Regression:
- **Linear activation** is used in the output layer to allow **any real number** as output.  
- **Loss Function:** Mean Squared Error (MSE) or Mean Absolute Error (MAE) is used to train the model.  
- **Feature Scaling:** Important to normalize the input values to improve model convergence.  

**Why ANN is suitable for regression:**
- Can model **non-linear relationships** between features and target  
- Learns **complex interactions** automatically  
- Scales well with large datasets and multiple features  

---

## 💡 How to Use the App
1. Open the **Streamlit app**.  
2. Enter customer details such as Credit Score, Age, Tenure, Balance, Gender, Geography, etc.  
3. Click **Predict Salary**.  
4. The app outputs the **predicted estimated salary**.  

---

## 📊 Results & Insights
- The ANN regression model predicts salaries based on customer features.  
- Key features influencing salary prediction include:  
  - Age  
  - Credit Score  
  - Balance  
  - Tenure  
  - Number of Products  

---

## 👨‍💻 Author
**Chandrashekhar D**  
💻 Data Science & Machine Learning Enthusiast  
📧 [chandrashekhard543@gmail.com]

---

## 🏁 Acknowledgments
- Churn Modelling dataset for realistic customer data  
- TensorFlow & Keras for ANN modeling  
- Streamlit for web app deployment  

---

## 📜 License
This project is open-source under the **MIT License**.
