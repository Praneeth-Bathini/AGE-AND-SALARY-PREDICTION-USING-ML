import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import tkinter as tk
from tkinter import filedialog

# Initialize variables for the model, data, and file path
model = None
data = None
file_path = None

# Function to load the CSV file
def load_csv_file():
    global data, file_path
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        data = pd.read_csv(file_path)
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, f"File Loaded: {file_path}\nReady to Train!")

# Function to train the model
def train_model():
    global model, data
    
    if data is None:
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, "Please load a CSV file first.")
        return
    
    # Splitting the data into features and target
    X = data[['Age']]
    y = data['Salary']
    
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Training the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Calculate and show accuracy in the result_text box
    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test, y_pred)
    
    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, f"Model Trained: LinearRegression\nAccuracy: {accuracy:.2f}")

# Function to predict salary
def predict_salary():
    if model is None:
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, "Please train the model first.")
        return
    
    try:
        age = float(age_entry.get())
        predicted_salary = model.predict([[age]])[0]
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, f"Predicted Salary: ${predicted_salary:.2f}")
    except ValueError:
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, "Please enter a valid age.")

# Create the Tkinter GUI
root = tk.Tk()
root.title("Salary Prediction")

# Widgets
load_button = tk.Button(root, text="Load CSV File", command=load_csv_file)
load_button.pack()

age_label = tk.Label(root, text="Enter Age:")
age_label.pack()

age_entry = tk.Entry(root)
age_entry.pack()

train_button = tk.Button(root, text="Train Model", command=train_model)
train_button.pack()

predict_button = tk.Button(root, text="Predict Salary", command=predict_salary)
predict_button.pack()

result_text = tk.Text(root, height=10, width=50)
result_text.pack()

# Run the application
root.mainloop()
