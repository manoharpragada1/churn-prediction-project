# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset - I'm really excited to dig into this data!
data = pd.read_csv('OnlineRetail.csv', encoding='latin1')  # 'latin1' to handle any special characters

# Drop rows with missing CustomerID - because we can't analyze customers without knowing who they are!
data_cleaned = data.dropna(subset=['CustomerID'])

# Convert CustomerID to string to avoid potential numeric issues
data_cleaned['CustomerID'] = data_cleaned['CustomerID'].astype(str)

# Convert InvoiceDate to datetime for better analysis
data_cleaned['InvoiceDate'] = pd.to_datetime(data_cleaned['InvoiceDate'])

# Create a TotalPrice column (Quantity * UnitPrice) - this gives us the total spend per transaction!
data_cleaned['TotalPrice'] = data_cleaned['Quantity'] * data_cleaned['UnitPrice']

# Let's filter out any transactions with negative quantities or zero price, which may be returns or errors.
data_cleaned = data_cleaned[(data_cleaned['Quantity'] > 0) & (data_cleaned['UnitPrice'] > 0)]

# Preview the cleaned data
print(data_cleaned.head())  # Wow, this looks much cleaner now!
# Set a reference date for recency calculation (let's take the day after the last transaction)
reference_date = data_cleaned['InvoiceDate'].max() + pd.Timedelta(days=1)

# Group by CustomerID to calculate RFM values
rfm = data_cleaned.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency: Days since the last purchase
    'InvoiceNo': 'nunique',  # Frequency: Number of unique purchases
    'TotalPrice': 'sum'  # Monetary: Total money spent
})

# Rename the columns for better understanding
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Let’s exclude customers with zero or negative monetary value (if any)
rfm = rfm[rfm['Monetary'] > 0]

# Preview the RFM data
print(rfm.describe())  # This summary gives us a great look at customer behavior!
# Define churn: If Recency is greater than 180 days, we'll consider the customer churned
rfm['Churn'] = (rfm['Recency'] > 180).astype(int)  # 1 = churned, 0 = not churned

# Let's see how many customers are churned vs. active
print(rfm['Churn'].value_counts())  # This will show us the distribution of churned customers!
# Features: Recency, Frequency, and Monetary
X = rfm[['Recency', 'Frequency', 'Monetary']]

# Target: Churn
y = rfm['Churn']

# Optional: Log-transform the Monetary value to reduce skewness (many customers have very small monetary values)
X['Monetary'] = np.log1p(X['Monetary'])  # This helps the model learn better with large value ranges
# Split the data (80% training, 20% testing) - this split allows us to validate our model!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier - I love how Random Forest works like an ensemble of decision trees!
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model’s performance with accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')  # Wow, I can’t wait to see how accurate our model is!
    # Confusion Matrix - this matrix shows where the model got things right and wrong
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()  # Visualizing results is one of my favorite parts!
