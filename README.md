<h1 align="center">
AgriPredictor: Crop Recommendation using Machine Learning
</h1>

---

## 📝 Abstract

<div align="justify">

AgriPredictor is a machine learning-based crop recommendation system designed to help farmers and agricultural stakeholders determine the most suitable crops based on soil and environmental parameters. Using a dataset of 2,200 entries, the system leverages machine learning techniques such as Random Forest, SVM, Logistic Regression, KNN, and Decision Tree to identify the best crop to cultivate under given conditions.

The model was trained using PyTorch and achieved a high classification accuracy of 99.24% with the Random Forest algorithm. The application demonstrates how intelligent systems can support efficient agricultural planning, improve productivity, and optimize resource usage through accurate predictions.

</div>

---

## 📁 Repository Structure

```bash
AgriPredictor/
├── data/
│   └── Crop_recommendation.csv       # Dataset with soil and climate features
├── model/
│   ├── baseline/                     # Pretrained PyTorch model
│   ├── pkl_files/                    # Label encoder for crop classes
│   └── normalization/                # Mean and standard deviation files
├── model_training.ipynb              # Jupyter notebook for data prep and training
├── app.py                            # Flask app for deployment
└── README.md                         # Project documentation
```
---
## 🎯 Key Features

- Cleaned and normalized dataset using MinMaxScaler  
- Multiple classification models trained and evaluated  
- PyTorch-based custom neural network model  
- Train-validation split with performance evaluation  
- Label encoding and model serialization for deployment  
- Random Forest model achieved 99.24% accuracy  

---

## 🛠️ Technologies Used

- **Programming Language**: Python 3.x  
- **Libraries**: NumPy, Pandas, Matplotlib, Scikit-learn, PyTorch, Pickle, Flask  
- **Development Tools**: Jupyter Notebook, VS Code  

---

## 📊 Dataset Description

The dataset contains 2,200 records with the following features:

- Nitrogen (N), Phosphorus (P), Potassium (K)  
- Temperature, Humidity, pH, Rainfall  
- Crop label (target variable)

The features were normalized and label-encoded before model training.

---

## 📈 Results Summary

| Model                        | Accuracy  |
|------------------------------|-----------|
| Logistic Regression          | ~94%      |
| Naive Bayes                  | ~93%      |
| Support Vector Machine (SVM)| ~96%      |
| K-Nearest Neighbors (KNN)    | ~97%      |
| Decision Tree                | ~97%      |
| **Random Forest**            | **99.24%** |

*Random Forest provided the best overall performance on the validation set.*
---
## 📬 Contact Information

For any detailed information, clarification, or collaboration inquiries regarding this project, feel free to reach out:

- **Email**: [manne.bharadwaj.1953@gmail.com](mailto:manne.bharadwaj.1953@gmail.com)
- **LinkedIn**: [Bharadwaj Manne](https://www.linkedin.com/in/bharadwaj-manne-711476249/)
