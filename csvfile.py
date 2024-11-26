# "import pandas as pd
# from sklearn.model_selection import train_test_split

# # Load the original dataset
# data = pd.read_csv('C:/Users/HP/Downloads/archive/creditcard.csv')

# # Split the data into features (X) and target (y)
# X = data.drop(columns=['Class'])
# y = data['Class']

# # Split the data into training and test sets (using stratification to maintain class balance)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# # Separate the test set into fraudulent and legitimate transactions
# fraudulent_transactions = X_test[y_test == 1]
# legitimate_transactions = X_test[y_test == 0]

# # Balance the dataset by taking an equal number of fraudulent and legitimate transactions
# num_samples = min(len(fraudulent_transactions), len(legitimate_transactions))
# balanced_fraudulent = fraudulent_transactions.sample(n=num_samples, random_state=42)
# balanced_legitimate = legitimate_transactions.sample(n=num_samples, random_state=42)

# # Concatenate the balanced fraudulent and legitimate transactions
# balanced_data = pd.concat([balanced_fraudulent, balanced_legitimate])

# # Add the target 'Class' back to the balanced dataset
# balanced_data['Class'] = y_test.loc[balanced_data.index]

# # Save the balanced data to a new CSV file
# balanced_data.to_csv('C:/Users/HP/Downloads/balanced_test_set.csv', index=False)

# print("Balanced CSV with fraud and legitimate transactions saved successfully!")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the original dataset
data = pd.read_csv('C:/Users/HP/Downloads/archive/creditcard.csv')

# Split the data into features (X) and target (y)
X = data.drop(columns=['Class'])
y = data['Class']

# Split the data into training and test sets (using stratification to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Separate the test set into fraudulent and legitimate transactions
fraudulent_transactions = X_test[y_test == 1]
legitimate_transactions = X_test[y_test == 0]

# Balance the dataset by taking an equal number of fraudulent and legitimate transactions
num_samples = min(len(fraudulent_transactions), len(legitimate_transactions))
balanced_fraudulent = fraudulent_transactions.sample(n=num_samples, random_state=42)
balanced_legitimate = legitimate_transactions.sample(n=num_samples, random_state=42)

# Concatenate the balanced fraudulent and legitimate transactions
balanced_data = pd.concat([balanced_fraudulent, balanced_legitimate])

# Add the target 'Class' back to the balanced dataset
balanced_data['Class'] = y_test.loc[balanced_data.index]

# Normalize the features using MinMaxScaler
scaler = MinMaxScaler()
feature_columns = balanced_data.drop(columns=['Class']).columns  # Exclude 'Class' for normalization
balanced_data[feature_columns] = scaler.fit_transform(balanced_data[feature_columns])

# Save the balanced and normalized data to a new CSV file
balanced_data.to_csv('C:/Users/HP/Downloads/balanced_test_set.csv', index=False)

print("Balanced and normalized CSV with fraud and legitimate transactions saved successfully!")
