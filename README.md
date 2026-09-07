# MetaRisk XAI: Clinical Explainable AI & Diagnostic Dashboard

An end-to-end, production-ready machine learning and Explainable AI (XAI) platform designed to assess metabolic syndrome and diabetes risk from clinical patient vitals. Built entirely in **Python**, this application bridges high-performance predictive modeling with rigorous medical interpretability (SHAP & LIME) and automated PDF report generation.

---

## Key Features

- **Automated Feature Engineering**: Computes clinical interaction terms including Glucose-BMI interaction, Insulin-Glucose ratios, Genetic-Age exposure, and Blood Pressure-Age ratios.
- **Optimized Machine Learning**: Powered by a robust Random Forest classifier benchmarked against Logistic Regression and Gradient Boosting models.
- **Explainable AI (XAI)**: 
  - **Global Interpretability**: SHAP summary plots highlighting feature importance across the dataset.
  - **Local Interpretability**: Individualized patient risk breakdowns using SHAP Force Plots and LIME explanations.
- **Interactive Pure-Python UI**: Built with **Streamlit** to deliver a responsive, zero-JavaScript dashboard with live inference sliders.
- **Automated Clinical Reporting**: Generates downloadable, publication-grade PDF diagnostic reports on the fly using **ReportLab**.

---

## Project Structure

```text
Explainable-AI-XAI-for-Metabolic-Syndrome-Risk/
│
├── main.py              # Core training script, evaluation, and XAI plot generation
├── app.py               # Pure-Python Streamlit UI and PDF generation engine
├── validation.py        # 10-fold cross-validation and benchmarking script
├── comparison.py        # Comparative model performance analysis
├── diabetes.csv         # Clinical dataset source
├── model.pkl            # Serialized trained Random Forest model
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation

```

## Installation & Setup
**Clone the Repository:**

**Bash**
git clone [https://github.com/iamarshiya/Explainable-AI-XAI-for-Metabolic-Syndrome-Risk.git](https://github.com/iamarshiya/Explainable-AI-XAI-for-Metabolic-Syndrome-Risk.git)
cd Explainable-AI-XAI-for-Metabolic-Syndrome-Risk
Create and Activate a Virtual Environment:
python -m venv venv
**On Windows:**
venv\Scripts\activate
**On macOS/Linux:**
source venv/bin/activate

Install Dependencies:
pip install pandas numpy scikit-learn matplotlib seaborn shap lime reportlab streamlit joblib

# How to Run the Application
Train and Serialize the Model:
Run the training pipeline to preprocess data, train the Random Forest model, output evaluation metrics, generate XAI plots, and 

serialize the model to model.pkl:
python main.py

Launch the Streamlit Dashboard:
Start the interactive web interface:
streamlit run app.py

Access the Web Interface:
Open your browser and navigate to http://localhost:8501. Adjust patient metrics in the sidebar, run the AI diagnostic, and download your automated PDF clinical report.

Model Performance & Benchmarking
The platform's underlying classification engine is evaluated using stratified cross-validation and holdout metrics:

**Algorithm:** Random Forest Classifier (Optimized via GridSearchCV)

**Validation Strategy:** 10-Fold Cross-Validation & Holdout Testing

**Metrics Tracked:** Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

## License
This project is licensed under the MIT License - see the LICENSE file for details.