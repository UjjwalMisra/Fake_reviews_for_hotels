# Fake Review Detection Project 🕵️‍♂️

A web application built with Python and Streamlit to classify hotel reviews as either **Genuine (Truthful)** or **Deceptive (Fake)**. This project uses a Multinomial Naive Bayes classifier trained on the Deceptive Opinion Spam Corpus.

## ✨ Features

-   **Real-time Prediction**: Enter any review text and get an instant classification.
-   **Confidence Score**: See the model's confidence in its prediction.
-   **Simple Web Interface**: Easy-to-use interface powered by Streamlit.
-   **Reproducible Model**: A clear training script to reproduce the machine learning model.

## 📸 Screenshot

 ![image](https://github.com/user-attachments/assets/dd33b6c8-4a7c-4aad-8aa5-d408af514f02)

![image](https://github.com/user-attachments/assets/9c17a676-509e-492a-b28a-716b06416a69)

![image](https://github.com/user-attachments/assets/9cf8c67f-28bf-4c56-a394-1d9cf0fa20a4)


---

## ⚙️ Technology Stack

-   **Backend & ML**: Python
-   **Web Framework**: Streamlit
-   **Data Manipulation**: Pandas, NumPy
-   **Machine Learning**: Scikit-learn

---
---

## 📂 Project Structure
Use code with caution.
Markdown
fake-review-project/
├── data/
│ └── deceptive-opinion.csv
├── .gitignore
├── app.py # The Streamlit web application
├── model_training.py # Script to train the model
├── model.pkl # Saved trained model
├── vectorizer.pkl # Saved CountVectorizer
├── requirements.txt # Python dependencies
└── README.md
