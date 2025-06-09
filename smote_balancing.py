import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('KaggleV2-May-2016.csv')

# Separate features and target
X = data.drop('No-show', axis=1)
y = data['No-show']

# Encode categorical features
for column in X.columns:
    if X[column].dtype == 'object':
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])

# Encode target variable: No = 0 (attended), Yes = 1 (no-show)
if y.dtype == 'object':
    y = y.map({'No': 0, 'Yes': 1})

# Apply SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Combine into a new balanced DataFrame
balanced_data = pd.concat(
    [pd.DataFrame(X_resampled, columns=X.columns),
     pd.Series(y_resampled, name='No-show')],
    axis=1
)

# Save the balanced dataset
balanced_data.to_csv('balanced_data.csv', index=False)

print("✅ SMOTE applied successfully. 'balanced_data.csv' saved.")
