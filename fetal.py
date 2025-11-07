import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

st.title('Fetal Health Classification: A Machine Learning App')
st.image('fetal_health_image.gif')
st.markdown("Use our advanced Machine Learning application to predict fetal health classification")

df = pd.read_csv("fetal_health.csv")

st.sidebar.header("Fetal Health Features Input")
st.sidebar.write('Upload your file')

uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

st.sidebar.warning("Ensure your data strictly follows the format outlined below.")
st.sidebar.dataframe(df.head())

model_choice = st.sidebar.radio("Select your model", ['Decision Tree', 'Random Forest', 'ADABoost','Soft Voting'])
st.sidebar.info(f"You selected: {model_choice}")

## used chatgpt 5.0 to help me understand having multiple model files and model path structure
model_files = {
    'Decision Tree': 'fetal_dt.pickle',
    'Random Forest': 'rf_fetal.pickle',
    'ADABoost': 'ada_fetal.pickle',
    'Soft Voting': 'soft_voting_fetal.pickle'
}

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File Uploaded!")

    model_path = model_files[model_choice]
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()

## used ChatGPT 5.0 to better format predicitions and it showed me how to use try and except
    try:
        preds = model.predict(df)
        probs = model.predict_proba(df)
        max_probs = np.max(probs, axis=1)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()
## used chatgpt 5.0 to understand label_map
    label_map = {1: 'Normal', 2: 'Suspect', 3: 'Pathological'}
    df['Fetal Health'] = [label_map[p] for p in preds]
    df['Prediction Probability (%)'] = np.round(max_probs * 100, 1)

## searched up how to highlight cells / background color
    def color_fetal(val):
        if val == 'Normal':
            return 'background-color: lime; color: black;'
        elif val == 'Suspect':
            return 'background-color: yellow; color: black;'
        elif val == 'Pathological':
            return 'background-color: orange; color: black;'
        else:
            return ''

    styled_df = df.style.applymap(color_fetal, subset=['Fetal Health'])

    st.subheader(f"Prediction using {model_choice} Model")
    st.dataframe(styled_df)

    st.title("Model Performance and Insights")

    if model_choice == 'Decision Tree':
        tab1, tab2, tab3 = st.tabs(["Feature Importance", "Classification Report", "Confusion Matrix"])

        # Tab 1: Feature Importance 
        with tab1:
            st.write("Feature Importance")
            st.image('feature_imp.svg')
            st.caption("Relative importance of features in prediction.")

        # Tab 2: Classification Report
        with tab2:
            st.write("### Classification Report")
            report_df = pd.read_csv('class_report.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")

        # Tab 3: Confusion Matrix
        with tab3:
            st.write("Confusion Matrix")
            st.image('confusion_mat.svg')
            st.caption("Confusion Matrix of model predictions.")

    elif model_choice == "Random Forest":
        tab1, tab2, tab3 = st.tabs(["Feature Importance", "Classification Report", "Confusion Matrix"])

        # Tab 1: Feature Importance 
        with tab1:
            st.write("Feature Importance")
            st.image('rf_feature_imp.svg')
            st.caption("Relative importance of features in prediction.")

        # Tab 2: Classification Report
        with tab2:
            st.write("### Classification Report")
            report_df = pd.read_csv('rf_class_report.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")


        # Tab 3: Confusion Matrix
        with tab3:
            st.write("Confusion Matrix")
            st.image('rf_confusion_mat.svg')
            st.caption("Confusion Matrix of model predictions.")

    elif model_choice == "ADABoost":
        tab1, tab2, tab3 = st.tabs(["Feature Importance", "Classification Report", "Confusion Matrix"])

        # Tab 1: Feature Importance 
        with tab1:
            st.write("Feature Importance")
            st.image('ada_feature_imp.svg')
            st.caption("Relative importance of features in prediction.")

        # Tab 2: Classification Report
        with tab2:
            st.write("### Classification Report")
            report_df = pd.read_csv('ada_class_report.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")

        # Tab 3: Confusion Matrix
        with tab3:
            st.write("Confusion Matrix")
            st.image('ada_confusion_mat.svg')
            st.caption("Confusion Matrix of model predictions.")

    elif model_choice == "Soft Voting":
        tab1, tab2, tab3 = st.tabs(["Feature Importance", "Classification Report", "Confusion Matrix"])

        # Tab 1: Feature Importance 
        with tab1:
            st.write("Feature Importance")
            st.image('soft_vote_feature_imp.svg')
            st.caption("Relative importance of features in prediction.")

        # Tab 2: Classification Report
        with tab2:
            st.write("### Classification Report")
            report_df = pd.read_csv('soft_vote_class_report.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")

        # Tab 3: Confusion Matrix
        with tab3:
            st.write("Confusion Matrix")
            st.image('soft_vote_confusion_mat.svg')
            st.caption("Confusion Matrix of model predictions.")

else:
    st.info("👈 Upload a CSV file from the sidebar to begin.")

