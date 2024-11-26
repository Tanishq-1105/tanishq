# from flask import Flask, render_template, request
# import pandas as pd
# import joblib
# from sklearn.metrics import accuracy_score, confusion_matrix

# # Load the trained model
# model = joblib.load('fraud_detection_model.pkl')

# # Load the balanced test set
# data = pd.read_csv('C:/Users/HP/Downloads/balanced_test_set.csv')

# # Initialize Flask app
# app = Flask(__name__)

# # Separate the features (X) and the target (y)
# X = data.drop(columns=['Class'])
# y = data['Class']

# # Get indices for the balanced test set transactions
# test_indices = X.index.tolist()

# # Evaluate the model on the balanced test set
# y_pred = model.predict(X)
# accuracy = accuracy_score(y, y_pred)
# conf_matrix = confusion_matrix(y, y_pred)

# @app.route('/')
# def home():
#     # Convert confusion matrix to string for display on the webpage
#     conf_matrix_str = str(conf_matrix)
#     return render_template('index.html', accuracy=accuracy, conf_matrix=conf_matrix_str, test_indices=test_indices)

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get the transaction number entered by the user
#         transaction_number = int(request.form['transaction_number'])
        
#         # Ensure the transaction number is valid (exists in the balanced test set)
#         if transaction_number not in test_indices:
#             return render_template('index.html', prediction_text="Invalid transaction number (not from the test set). Please choose a valid transaction.", accuracy=accuracy, conf_matrix=str(conf_matrix), test_indices=test_indices)
        
#         # Get the transaction details from the balanced test set
#         transaction = [X.iloc[transaction_number]]
        
#         # Predict fraud
#         prediction = model.predict(transaction)
#         result = "Fraudulent Transaction Detected!" if prediction == 1 else "Transaction is Legitimate."
        
#         # Convert confusion matrix to string for display
#         conf_matrix_str = str(conf_matrix)
        
#         return render_template('index.html', prediction_text=result, accuracy=accuracy, conf_matrix=conf_matrix_str, test_indices=test_indices)
    
#     except ValueError:
#         # If user enters a non-integer value, display error message
#         return render_template('index.html', prediction_text="Please enter a valid transaction number.", accuracy=accuracy, conf_matrix=str(conf_matrix), test_indices=test_indices)

# if __name__ == "__main__":
#     app.run(debug=True)
from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix

# Initialize Flask app
app = Flask(__name__)

# Load all models into a dictionary
models = {
    'random_forest': joblib.load('fraud_detection_model.pkl'),  # Placeholder for Random Forest
    'knn': joblib.load('fraud_detection_model.pkl'),  # Placeholder for KNN model
    'classification_model': joblib.load('fraud_detection_model.pkl')  # Placeholder for Classification Model
}

# Load the balanced test set
data = pd.read_csv('C:/Users/HP/Downloads/balanced_test_set.csv')

# Separate features (X) and the target (y)
X = data.drop(columns=['Class'])
y = data['Class']

# Evaluate each model and store metrics
model_metrics = {}
for name, model in models.items():
    y_pred = model.predict(X)
    model_metrics[name] = {
        'accuracy': accuracy_score(y, y_pred),
        'conf_matrix': confusion_matrix(y, y_pred).tolist()  # Convert confusion matrix to list for rendering
    }

@app.route('/')
def home():
    return render_template('index.html', metrics=None, download_link=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the transaction number and classifier choice from the form
        transaction_number = int(request.form['transaction_number'])
        classifier_choice = request.form['classifier_choice']
        
        # Check if classifier exists
        if classifier_choice not in models:
            return render_template('index.html', prediction_text="Invalid classifier selected.", metrics=None, download_link=None)
        
        # Ensure transaction number is valid
        if transaction_number not in X.index:
            return render_template('index.html', prediction_text="Invalid transaction number. Please choose a valid one.", metrics=None, download_link=None)
        
        # Get the model and make prediction
        model = models[classifier_choice]
        transaction = [X.iloc[transaction_number]]
        prediction = model.predict(transaction)
        result = "Fraudulent Transaction Detected!" if prediction == 1 else "Transaction is Legitimate."
        
        # Get the model metrics to display
        model_result = model_metrics[classifier_choice]
        
        # Provide download link for normalized CSV
        download_link = '/download_csv'
        
        return render_template('index.html', prediction_text=result, metrics=model_result, classifier_choice=classifier_choice, download_link=download_link)
    
    except ValueError:
        return render_template('index.html', prediction_text="Please enter a valid transaction number.", metrics=None, download_link=None)

@app.route('/download_csv')
def download_csv():
    # Provide normalized CSV file for download
    path = 'C:/Users/HP/Downloads/balanced_test_set.csv'
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
