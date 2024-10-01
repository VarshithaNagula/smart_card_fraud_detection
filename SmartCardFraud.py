import tkinter as tk
from tkinter import messagebox, filedialog
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import time
import pickle
import joblib
import tkinter as tk
from tkinter import simpledialog, messagebox

# Initialize the main window
main = tk.Tk()
main.title("Smart Card Fraud Detection using ensemble methods in Machine Learning")
main.geometry("1400x1200")

global filename
global dataset
global X_train, X_test, y_train, y_test
global table_output
global metrics_rf, metrics_combined

# Function to upload dataset
def uploadDataset():
    global filename, dataset
    filename = filedialog.askopenfilename(initialdir="dataset")
    text.delete('1.0', tk.END)
    text.insert(tk.END, f"{filename} loaded\n")
    dataset = pd.read_csv(filename)
    text.insert(tk.END, str(dataset.head()))
    text.insert(tk.END, f"Dataset columns: {list(dataset.columns)}\n")

# Function to preprocess dataset
def preprocessDataset():
    global X_train, X_test, y_train, y_test, dataset
    text.delete('1.0', tk.END)
    
    # Defining features and labels
    X = dataset.drop(['isFraud', 'isFlaggedFraud', 'nameOrig', 'nameDest'], axis=1)
    y = dataset['isFraud']
    
    # Identifying categorical and numerical columns
    categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]
    numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
    
    # Defining the preprocessing steps for numerical and categorical features
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Combining the preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Creating a pipeline that combines preprocessing with model training
    clf = Pipeline(steps=[('preprocessor', preprocessor)])

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing the data
    X_train = clf.fit_transform(X_train)
    X_test = clf.transform(X_test)

    # Save the preprocessing pipeline
    with open('model/preprocessing_pipeline.pkl', 'wb') as file:
        pickle.dump(clf.named_steps['preprocessor'], file)
    
    text.insert(tk.END, "Dataset Preprocessed\n")
    text.insert(tk.END, f"Training records: {X_train.shape[0]}\n")
    text.insert(tk.END, f"Testing records: {X_test.shape[0]}\n")

def calculateMetrics(predict, trueValue, name):
    report = classification_report(trueValue, predict, output_dict=True)
    accuracy = report['accuracy']
    precision = report['1']['precision']
    recall = report['1']['recall']
    f1_score = report['1']['f1-score']
    text.insert(tk.END, f"{name} Accuracy: {accuracy}\n")
    text.insert(tk.END, f"{name} Precision: {precision}\n")
    text.insert(tk.END, f"{name} Recall: {recall}\n")
    text.insert(tk.END, f"{name} F1-Score: {f1_score}\n")
    return report

def buildCombinedModel(algorithm, file_name):
    if os.path.exists('model/' + file_name):
        with open('model/' + file_name, 'rb') as file:
            algorithm = pickle.load(file)
    else:
        start_time = time.time()
        algorithm.fit(X_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        with open('model/' + file_name, 'wb') as file:
            pickle.dump(algorithm, file)
    predict = algorithm.predict(X_test)
    return predict

def trainRF():
    global metrics_rf
    text.delete('1.0', tk.END)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)  # Example parameters, adjust as needed
    predict = buildCombinedModel(rf, "rf_model.pkl")
    metrics_rf = calculateMetrics(predict, y_test, "Random Forest")
    return metrics_rf

def trainCombinedModel():
    global metrics_combined
    text.delete('1.0', tk.END)
    rf = RandomForestClassifier()
    dt = DecisionTreeClassifier()
    combined_model = VotingClassifier(estimators=[('rf', rf), ('dt', dt)], voting='hard')
    
    predict = buildCombinedModel(combined_model, "combined_rf_dt.pkl")
    metrics_combined = calculateMetrics(predict, y_test, "Combined RandomForest + DecisionTree")
    return metrics_combined


# Function to preprocess input values
def preprocess_input(input_df):
    # Load your preprocessing pipeline
    with open('model/preprocessing_pipeline.pkl', 'rb') as file:
        pipeline = pickle.load(file)
    # Transform the input values
    preprocessed_values = pipeline.transform(input_df)
    return preprocessed_values
'''
def predict_fraud():
    try:
        # Input dialogs for user to manually enter feature values
        feature_values = {
            "step": simpledialog.askinteger("Input", "Enter step:"),
            "type": simpledialog.askstring("Input", "Enter type (PAYMENT, TRANSFER, CASH_OUT, DEBIT):"),
            "amount": simpledialog.askfloat("Input", "Enter amount:"),
            "oldbalanceOrg": simpledialog.askfloat("Input", "Enter old balance of origin account:"),
            "newbalanceOrig": simpledialog.askfloat("Input", "Enter new balance of origin account:"),
            "oldbalanceDest": simpledialog.askfloat("Input", "Enter old balance of destination account:"),
            "newbalanceDest": simpledialog.askfloat("Input", "Enter new balance of destination account:")
        }

        # Create a DataFrame from input values
        input_df = pd.DataFrame([feature_values])

        # Preprocess the input values
        preprocessed_input = preprocess_input(input_df)

        # Load the combined model
        combined_model = joblib.load('model/combined_rf_dt.pkl')

        # Predict using the combined model
        combined_predictions = combined_model.predict(preprocessed_input)

        # Display results
        text.delete('1.0', tk.END)
        result_combined = "Fraud" if combined_predictions[0] == 1 else "Not Fraud"
        input_values_str = ", ".join([f"{key}: {value}" for key, value in feature_values.items()])
        text.insert(tk.END, f"Input Values: {input_values_str} -> Combined Model Prediction: {result_combined}\n")

    except Exception as e:
        messagebox.showerror("Prediction Error", "An error occurred during prediction.")
        print(e)'''

def predict_fraud():
    try:
        # Input dialogs for user to manually enter feature values
        feature_values = {
            "step": simpledialog.askinteger("Input", "Enter step:"),
            "type": simpledialog.askstring("Input", "Enter type (PAYMENT, TRANSFER, CASH_OUT, DEBIT):"),
            "amount": simpledialog.askfloat("Input", "Enter amount:"),
            "oldbalanceOrg": simpledialog.askfloat("Input", "Enter old balance of origin account:"),
            "newbalanceOrig": simpledialog.askfloat("Input", "Enter new balance of origin account:"),
            "oldbalanceDest": simpledialog.askfloat("Input", "Enter old balance of destination account:"),
            "newbalanceDest": simpledialog.askfloat("Input", "Enter new balance of destination account:"),
            "nameOrig": simpledialog.askstring("Input", "Enter origin account name :"),
            "nameDest": simpledialog.askstring("Input", "Enter destination account name:")
        }

        # Create a DataFrame from input values
        input_df = pd.DataFrame([feature_values])

        # Preprocess the input values (assuming preprocessing does not alter display columns)
        preprocessed_input = preprocess_input(input_df)

        # Load the combined model
        combined_model = joblib.load('model/combined_rf_dt.pkl')

        # Predict using the combined model
        combined_predictions = combined_model.predict(preprocessed_input)

        # Display results
        text.delete('1.0', tk.END)
        result_combined = "Fraud" if combined_predictions[0] == 1 else "Not Fraud"

        # Update input values string to include nameOrig and nameDest
        input_values_str = ", ".join([f"{key}: {value}" for key, value in feature_values.items()])
        text.insert(tk.END, f"Input Values: {input_values_str} -> Combined Model Prediction: {result_combined}\n")

    except Exception as e:
        messagebox.showerror("Prediction Error", "An error occurred during prediction.")
        print(e)


def graph():
    global metrics_rf, metrics_combined
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    rf_scores = [metrics_rf['accuracy'], metrics_rf['1']['precision'], metrics_rf['1']['recall'], metrics_rf['1']['f1-score']]
    combined_scores = [metrics_combined['accuracy'], metrics_combined['1']['precision'], metrics_combined['1']['recall'], metrics_combined['1']['f1-score']]

    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x, rf_scores, width, label='Random Forest')
    ax.bar([p + width for p in x], combined_scores, width, label='Combined RF+DT')

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Model Comparison')
    ax.set_xticks([p + width/2 for p in x])
    ax.set_xticklabels(metrics)
    ax.legend()

    plt.show()

# GUI Components
font = ('times', 16, 'bold')
title = tk.Label(main, text='Smart Card Fraud Detection using ensemble methods in Machine Learning')
title.config(bg='greenyellow', fg='dodger blue', font=font, height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = tk.Text(main, height=20, width=150)
scroll = tk.Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

uploadButton = tk.Button(main, text="Upload smart Card Dataset", command=uploadDataset)
uploadButton.place(x=50, y=550)
uploadButton.config(font=font1)

preprocessButton = tk.Button(main, text="Dataset Preprocessing", command=preprocessDataset)
preprocessButton.place(x=350, y=550)
preprocessButton.config(font=font1)

trainButton = tk.Button(main, text="Train Combined Model", command=trainCombinedModel)
trainButton.place(x=550, y=550)
trainButton.config(font=font1)

trainRFButton = tk.Button(main, text="Train RF Model", command=trainRF)
trainRFButton.place(x=750, y=550)
trainRFButton.config(font=font1)

predictButton = tk.Button(main, text="Predict Fraud", command=predict_fraud)
predictButton.place(x=950, y=550)
predictButton.config(font=font1)

graphButton = tk.Button(main, text="Graph Performance", command=graph)
graphButton.place(x=1150, y=550)
graphButton.config(font=font1)

main.config(bg='burlywood2')
main.mainloop()
