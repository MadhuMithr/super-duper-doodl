import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the dataset
file_path = r'C:\Users\MADHU MITHRA\Downloads\house-prices-advanced-regression-techniques\train.csv'
df = pd.read_csv(file_path)

# Assume X and y are already defined (features and target variable)
X = df[['LotArea', 'BedroomAbvGr', 'FullBath']]
y = df['SalePrice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate R^2 score
r2 = r2_score(y_test, y_pred)
print(f"Initial R^2 Score: {r2}")

# Example of adding more features or performing feature engineering
# X = df[['LotArea', 'BedroomAbvGr', 'FullBath', 'OverallQual', 'YearBuilt']]

# Retrain the model with additional features or different approach

# Make predictions again and calculate new R^2 score

# Evaluate other strategies mentioned above to improve the model further

# Example of cross-validation
# from sklearn.model_selection import cross_val_score
# cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
# print(f"Cross-validated R^2 scores: {cv_scores}")
# print(f"Average R^2 score: {cv_scores.mean()}")
