#https://catalog.data.gov/dataset/nchs-births-and-general-fertility-rates-united-states
import pandas as pd

df = pd.read_csv('/content/NCHS_-_Births_and_General_Fertility_Rates__United_States.csv')
print(df)

#https://www.cdc.gov/nchs/pressroom/sosmap/fertility_rate/fertility_rates.htm
import pandas as pd

df = pd.read_csv('/content/data-table.csv')
print(df)

import os             # Good for navigating your computer's files
import numpy as np    # Great for lists (arrays) of numbers
import pandas as pd   # Great for tables (google spreadsheets, microsoft excel, csv)
from sklearn.metrics import accuracy_score

data_path = '/content/data-table.csv'

# Use the 'pd.read_csv(filepath)' function to read in read our data and store it
# in a variable called 'dataframe'
dataframe = pd.read_csv(data_path)

# Redefine `dataframe` to include only the columns discussed
dataframe = dataframe[['YEAR', 'STATE', 'BIRTHS', 'FERTILITY RATE']]

dataframe.head()

dataframe.info()

import seaborn as sns
import matplotlib.pyplot as plt

sns.catplot(x = 'YEAR', y = 'BIRTHS', data = dataframe)
dataframe.head()

import os             # Good for navigating your computer's files
import numpy as np    # Great for lists (arrays) of numbers
import pandas as pd   # Great for tables (google spreadsheets, microsoft excel, csv)
from sklearn.metrics import accuracy_score

data_path = '/content/NationalAndStatePregnancy_PublicUse.csv'

# Use the 'pd.read_csv(filepath)' function to read in read our data and store it
# in a variable called 'dataframe'
dataframe = pd.read_csv(data_path)

# Redefine `dataframe` to include only the columns discussed
dataframe = dataframe[['state', 'year', 'birthratetotal', 'abortionratetotal', 'pregnancyratetotal', 'miscarriagestotal', 'birthstotal', 'abortionstotal']]

dataframe.head()

dataframe.info()

import seaborn as sns
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.pyplot as plt

# Use Seaborn's catplot with custom figure size and spacing
sns.catplot(
    x='year', y='abortionratetotal', data=dataframe,
    height=8, aspect=3, kind='point', dodge=True
)

sns.catplot(
    x='year', y='pregnancyratetotal', data=dataframe,
    height=8, aspect=3, kind='point', dodge=True
)

sns.catplot(
    x='year', y='miscarriagestotal', data=dataframe,
    height=8, aspect=3, kind='point', dodge=True
)

# Display the head of the DataFrame
dataframe.head()

print("Column names:")
print(dataframe.columns)
print("\nFirst few rows of data:")
print(dataframe.head())

import pandas as pd
import os             # Good for navigating your computer's files
import numpy as np    # Great for lists (arrays) of numbers
import pandas as pd   # Great for tables (google spreadsheets, microsoft excel, csv)
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
#https://catalog.data.gov/dataset/nchs-births-and-general-fertility-rates-united-states

df = pd.read_csv('/content/NCHS_-_Births_and_General_Fertility_Rates__United_States.csv')
print(df)
#https://www.cdc.gov/nchs/pressroom/sosmap/fertility_rate/fertility_rates.htm

df = pd.read_csv('/content/data-table.csv')
print(df)
data_path = '/content/data-table.csv'

# Use the 'pd.read_csv(filepath)' function to read in read our data and store it
# in a variable called 'dataframe'
dataframe = pd.read_csv(data_path)

# Redefine `dataframe` to include only the columns discussed
dataframe = dataframe[['YEAR', 'STATE', 'BIRTHS', 'FERTILITY RATE']]
dataframe.head()
dataframe.info()
sns.catplot(x = 'YEAR', y = 'BIRTHS', data = dataframe)
dataframe.head()
#https://osf.io/duj6a

df = pd.read_csv('/content/NationalAndStatePregnancy_PublicUse.csv')
print(df)
data_path = '/content/NationalAndStatePregnancy_PublicUse.csv'

# Use the 'pd.read_csv(filepath)' function to read in read our data and store it
# in a variable called 'dataframe'
dataframe = pd.read_csv(data_path)

# Redefine `dataframe` to include only the columns discussed
dataframe = dataframe[['state', 'year', 'birthratetotal', 'abortionratetotal', 'pregnancyratetotal', 'miscarriagestotal', 'birthstotal', 'abortionstotal']]
dataframe.head()
dataframe.info()

# Use Seaborn's catplot with custom figure size and spacing
sns.catplot(
    x='year', y='abortionratetotal', data=dataframe,
    height=8, aspect=3, kind='point', dodge=True  # Increase aspect for width, height for vertical space
)

sns.catplot(
    x='year', y='pregnancyratetotal', data=dataframe,
    height=8, aspect=3, kind='point', dodge=True  # Increase aspect for width, height for vertical space
)

sns.catplot(
    x='year', y='miscarriagestotal', data=dataframe,
    height=8, aspect=3, kind='point', dodge=True  # Increase aspect for width, height for vertical space
)

# Display the head of the DataFrame
dataframe.head()

# Create a mapping of states to numbers
state_to_number = {state: i for i, state in enumerate(dataframe['state'].unique())}

# Add a new column to the dataframe with the state number
dataframe['state_number'] = dataframe['state'].map(state_to_number)

# Print the updated dataframe
dataframe

# Install XGBoost package
!pip install xgboost>=1.6.0
!pip install shap
import shap

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score
from numpy import absolute


# Define target column
target_column = 'birthstotal'

# Assuming 'dataframe' is already defined somewhere in your code

# Data cleaning steps
# Convert categorical columns to numeric if any exist
dataframe = pd.get_dummies(dataframe, drop_first=True)  # One-hot encode categorical columns

# Handle missing values by dropping rows with NaNs
dataframe = dataframe.dropna()

# Ensure all columns are numeric
dataframe = dataframe.apply(pd.to_numeric, errors='coerce')
dataframe = dataframe.dropna()  # Drop rows with NaNs after conversion

# Define features and target
X = dataframe.drop(columns=[target_column])
y = dataframe[target_column]

# Convert X and y to NumPy arrays for compatibility
X = X.values
y = y.values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reconfirm data types and handle non-numeric types
X_train = np.array(X_train, dtype=np.float64)
y_train = np.array(y_train, dtype=np.float64)

# Define the model
model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)

# Fit the model on the training data
model.fit(X_train, y_train)

# Define model evaluation method for cross-validation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# Evaluate model with cross-validation
scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
scores = absolute(scores)  # Force scores to be positive

# Print the Mean Absolute Error (MAE) from cross-validation
print('Cross-validated Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))

# Test set evaluation
y_pred = model.predict(X_test)

# Calculate evaluation metrics on test data
test_mae = mean_absolute_error(y_test, y_pred)
test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
X_display = pd.DataFrame(X, columns=dataframe.drop(columns=[target_column]).columns)
shap.force_plot(explainer.expected_value, shap_values[0, :], X_display.iloc[0, :])
shap.force_plot(explainer.expected_value, shap_values[:1000, :], X_display.iloc[:1000, :])
shap.summary_plot(shap_values, X_display, plot_type="bar")
for name in X_display.columns:
    shap.dependence_plot(name, shap_values, X, display_features=X_display)

# Print test set metrics
print(f'Test MAE: {test_mae:.3f}')
print(f'Test MSE: {test_mse:.3f}')
print(f'Test R² Score: {test_r2:.3f}')

# Ensure all previous data preprocessing and cleaning steps are completed

# Define target column
target_column = 'birthstotal'

# Prepare data
X, y, feature_names = prepare_data(dataframe, target_column)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = train_xgboost_model(X_train, y_train)

# Evaluate the model
metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

# Print metrics
print("Model Performance Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Prepare display dataframe
X_display = pd.DataFrame(X, columns=feature_names)

# Generate and display SHAP plots
plot_shap_analysis(model, X, X_display, feature_names)

print("Column names:")
print(dataframe.columns)
print("\nFirst few rows of data:")
print(dataframe.head())

!pip install prophet  # Install the 'prophet' library

from prophet import Prophet  # Import Prophet from the 'prophet' library
import pandas as pd
import matplotlib.pyplot as plt

# Load the original data and filter for California
data_path = '/content/NationalAndStatePregnancy_PublicUse.csv'
dataframe = pd.read_csv(data_path)

# Filter data for California
state_name = 'CA'
if state_name in dataframe['state'].values:
    california_data = dataframe[dataframe['state'] == state_name].copy()
    print(len(california_data))
    print(len(dataframe))
    # Ensure 'year' column is in datetime format
    california_data['year'] = pd.to_datetime(california_data['year'], format='%Y')

    # Prepare data for Prophet - rename columns to 'ds' and 'y' for compatibility
    forecast_data = california_data[['year', 'birthstotal']].rename(columns={'year': 'ds', 'birthstotal': 'y'})

    # Initialize and fit the Prophet model
    model = Prophet(yearly_seasonality=True)
    model.fit(forecast_data)

    # Generate future dates (up to 2024)
    future = model.make_future_dataframe(periods=11, freq='Y')  # 10 more years after 2020

    # Predict future values
    forecast = model.predict(future)

    # Filter for forecasted years 2021-2024
    forecasted_years = forecast[forecast['ds'].dt.year.isin([2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030])]

    # Plot the forecast
    model.plot(forecast)
    plt.title(f"birthstotal Forecast for {state_name} (till 2030)")
    plt.xlabel("Year")
    plt.ylabel("birthstotal Total")
    plt.show()

    # Display forecasted data for the selected years
    print(forecasted_years[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
else:
    print(f"State '{state_name}' not found in the dataset.")

from sklearn.metrics import mean_squared_error
import numpy as np

# Split data into train/test
train = forecast_data[forecast_data['ds'].dt.year <= 2015]
test = forecast_data[forecast_data['ds'].dt.year > 2015]

# Fit Prophet on training data
model = Prophet(yearly_seasonality=True)
model.fit(train)

# Predict on test period
future_test = model.make_future_dataframe(periods=len(test), freq='Y')
forecast_test = model.predict(future_test)

# Merge actual vs predicted
forecast_test = forecast_test.set_index('ds').join(test.set_index('ds'))
forecast_test = forecast_test.dropna()

# Compute RMSE and MAPE
rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
mape = np.mean(np.abs((merged['y'] - merged['yhat']) / merged['y'])) * 100

print(f"Prophet RMSE: {rmse:.2f}")
print(f"Prophet MAPE: {mape:.2f}%")

!pip install prophet

from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
df = pd.read_csv('/content/NationalAndStatePregnancy_PublicUse.csv')
state_name = 'CA'
california_data = df[df['state'] == state_name].copy()
california_data['year'] = pd.to_datetime(california_data['year'], format='%Y')

# Prepare forecast_data
forecast_data = california_data[['year','birthstotal']].rename(columns={'year':'ds','birthstotal':'y'}).copy()

# Split train/test
train = forecast_data[forecast_data['ds'].dt.year <= 2015].copy()
test = forecast_data[forecast_data['ds'].dt.year > 2015].copy()

# Prophet
prophet_model = Prophet(yearly_seasonality=True)
prophet_model.fit(train)
future = prophet_model.make_future_dataframe(periods=len(test), freq='Y')
forecast = prophet_model.predict(future).copy()

# Merge on year
forecast['year'] = forecast['ds'].dt.year
test['year'] = test['ds'].dt.year
merged_prophet = pd.merge(test, forecast, on='year', how='inner')

# RMSE & MAPE
rmse_prophet = np.sqrt(mean_squared_error(merged_prophet['y'], merged_prophet['yhat']))
mape_prophet = np.mean(np.abs((merged_prophet['y'] - merged_prophet['yhat']) / merged_prophet['y'])) * 100
print(f"Prophet RMSE: {rmse_prophet:.2f}")
print(f"Prophet MAPE: {mape_prophet:.2f}%")

# Linear Regression
train['year_num'] = train['ds'].dt.year
test['year_num'] = test['ds'].dt.year
lr_model = LinearRegression()
lr_model.fit(train[['year_num']], train['y'])
test['y_pred_lr'] = lr_model.predict(test[['year_num']])
rmse_lr = np.sqrt(mean_squared_error(test['y'], test['y_pred_lr']))
mape_lr = np.mean(np.abs((test['y'] - test['y_pred_lr']) / test['y'])) * 100
print(f"Linear Regression RMSE: {rmse_lr:.2f}")
print(f"Linear Regression MAPE: {mape_lr:.2f}%")

# Plot
plt.figure(figsize=(10,6))
plt.plot(train['ds'], train['y'], label='Train', marker='o')
plt.plot(test['ds'], test['y'], label='Actual Test', marker='o')
plt.plot(merged_prophet['year'], merged_prophet['yhat'], label='Prophet Forecast', marker='x')
plt.plot(test['ds'], test['y_pred_lr'], label='Linear Regression Forecast', marker='s')
plt.xlabel('Year')
plt.ylabel('Total Births')
plt.title(f'Total Births Forecast for {state_name}')
plt.legend()
plt.grid(True)
plt.show()

!pip install prophet  # Install the 'prophet' library

from prophet import Prophet  # Import Prophet from the 'prophet' library
import pandas as pd
import matplotlib.pyplot as plt

# Load the original data and filter for Texas
data_path = '/content/NationalAndStatePregnancy_PublicUse.csv'
dataframe = pd.read_csv(data_path)

# Filter data for Texas
state_name = 'TX'
if state_name in dataframe['state'].values:
    texas_data = dataframe[dataframe['state'] == state_name].copy()
    print(len(texas_data))
    print(len(dataframe))
    # Ensure 'year' column is in datetime format
    texas_data['year'] = pd.to_datetime(texas_data['year'], format='%Y')

    # Prepare data for Prophet - rename columns to 'ds' and 'y' for compatibility
    forecast_data = texas_data[['year', 'birthstotal']].rename(columns={'year': 'ds', 'birthstotal': 'y'})

    # Initialize and fit the Prophet model
    model = Prophet(yearly_seasonality=True)
    model.fit(forecast_data)

    # Generate future dates (up to 2030)
    future = model.make_future_dataframe(periods=11, freq='Y')  # 10 more years after 2020

    # Predict future values
    forecast = model.predict(future)

    # Filter for forecasted years 2021-2030
    forecasted_years = forecast[forecast['ds'].dt.year.isin([2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030])]

    # Plot the forecast
    model.plot(forecast)
    plt.title(f"birthstotal Forecast for {state_name} (till 2030)")
    plt.xlabel("Year")
    plt.ylabel("birthstotal Total")
    plt.show()

    # Display forecasted data for the selected years
    print(forecasted_years[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
else:
    print(f"State '{state_name}' not found in the dataset.")

!pip install prophet

from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
df = pd.read_csv('/content/NationalAndStatePregnancy_PublicUse.csv')
state_name = 'TX'
texas_data = df[df['state'] == state_name].copy()
texas_data['year'] = pd.to_datetime(texas_data['year'], format='%Y')

# Prepare forecast_data
forecast_data = texas_data[['year','birthstotal']].rename(columns={'year':'ds','birthstotal':'y'}).copy()

# Split train/test
train = forecast_data[forecast_data['ds'].dt.year <= 2015].copy()
test = forecast_data[forecast_data['ds'].dt.year > 2015].copy()

# Prophet
prophet_model = Prophet(yearly_seasonality=True)
prophet_model.fit(train)
future = prophet_model.make_future_dataframe(periods=len(test), freq='Y')
forecast = prophet_model.predict(future).copy()

# Merge on year
forecast['year'] = forecast['ds'].dt.year
test['year'] = test['ds'].dt.year
merged_prophet = pd.merge(test, forecast, on='year', how='inner')

# RMSE & MAPE
rmse_prophet = np.sqrt(mean_squared_error(merged_prophet['y'], merged_prophet['yhat']))
mape_prophet = np.mean(np.abs((merged_prophet['y'] - merged_prophet['yhat']) / merged_prophet['y'])) * 100
print(f"Prophet RMSE: {rmse_prophet:.2f}")
print(f"Prophet MAPE: {mape_prophet:.2f}%")

# Linear Regression
train['year_num'] = train['ds'].dt.year
test['year_num'] = test['ds'].dt.year
lr_model = LinearRegression()
lr_model.fit(train[['year_num']], train['y'])
test['y_pred_lr'] = lr_model.predict(test[['year_num']])
rmse_lr = np.sqrt(mean_squared_error(test['y'], test['y_pred_lr']))
mape_lr = np.mean(np.abs((test['y'] - test['y_pred_lr']) / test['y'])) * 100
print(f"Linear Regression RMSE: {rmse_lr:.2f}")
print(f"Linear Regression MAPE: {mape_lr:.2f}%")

# Plot
plt.figure(figsize=(10,6))
plt.plot(train['ds'], train['y'], label='Train', marker='o')
plt.plot(test['ds'], test['y'], label='Actual Test', marker='o')
plt.plot(merged_prophet['year'], merged_prophet['yhat'], label='Prophet Forecast', marker='x')
plt.plot(test['ds'], test['y_pred_lr'], label='Linear Regression Forecast', marker='s')
plt.xlabel('Year')
plt.ylabel('Total Births')
plt.title(f'Total Births Forecast for {state_name}')
plt.legend()
plt.grid(True)
plt.show()

!pip install prophet  # Install the 'prophet' library

from prophet import Prophet  # Import Prophet from the 'prophet' library
import pandas as pd
import matplotlib.pyplot as plt

# Load the original data and filter for California
data_path = '/content/NationalAndStatePregnancy_PublicUse.csv'
dataframe = pd.read_csv(data_path)

# Filter data for Texas
state_name = 'CA'
if state_name in dataframe['state'].values:
    california_data = dataframe[dataframe['state'] == state_name].copy()
    print(len(california_data))
    print(len(dataframe))
    # Ensure 'year' column is in datetime format
    california_data['year'] = pd.to_datetime(california_data['year'], format='%Y')

    # Prepare data for Prophet - rename columns to 'ds' and 'y' for compatibility
    forecast_data = california_data[['year', 'abortionstotal']].rename(columns={'year': 'ds', 'abortionstotal': 'y'})

    # Initialize and fit the Prophet model
    model = Prophet(yearly_seasonality=True)
    model.fit(forecast_data)

    # Generate future dates (up to 2030)
    future = model.make_future_dataframe(periods=11, freq='Y')  # 10 more years after 2020

    # Predict future values
    forecast = model.predict(future)

    # Filter for forecasted years 2021-2030
    forecasted_years = forecast[forecast['ds'].dt.year.isin([2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030])]

    # Plot the forecast
    model.plot(forecast)
    plt.title(f"abortionstotal Forecast for {state_name} (till 2030)")
    plt.xlabel("Year")
    plt.ylabel("abortionstotal Total")
    plt.show()

    # Display forecasted data for the selected years
    print(forecasted_years[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
else:
    print(f"State '{state_name}' not found in the dataset.")

!pip install prophet  # Install the 'prophet' library

from prophet import Prophet  # Import Prophet from the 'prophet' library
import pandas as pd
import matplotlib.pyplot as plt

# Load the original data and filter for California
data_path = '/content/NationalAndStatePregnancy_PublicUse.csv'
dataframe = pd.read_csv(data_path)

# Filter data for Texas
state_name = 'TX'
if state_name in dataframe['state'].values:
    texas_data = dataframe[dataframe['state'] == state_name].copy()
    print(len(texas_data))
    print(len(dataframe))
    # Ensure 'year' column is in datetime format
    texas_data['year'] = pd.to_datetime(texas_data['year'], format='%Y')

    # Prepare data for Prophet - rename columns to 'ds' and 'y' for compatibility
    forecast_data = texas_data[['year', 'abortionstotal']].rename(columns={'year': 'ds', 'abortionstotal': 'y'})

    # Initialize and fit the Prophet model
    model = Prophet(yearly_seasonality=True)
    model.fit(forecast_data)

    # Generate future dates (up to 2030)
    future = model.make_future_dataframe(periods=11, freq='Y')  # 10 more years after 2020

    # Predict future values
    forecast = model.predict(future)

    # Filter for forecasted years 2021-2030
    forecasted_years = forecast[forecast['ds'].dt.year.isin([2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030])]

    # Plot the forecast
    model.plot(forecast)
    plt.title(f"abortionstotal Forecast for {state_name} (till 2030)")
    plt.xlabel("Year")
    plt.ylabel("abortionstotal Total")
    plt.show()

    # Display forecasted data for the selected years
    print(forecasted_years[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
else:
    print(f"State '{state_name}' not found in the dataset.")

!pip install prophet  # Install the 'prophet' library

from prophet import Prophet  # Import Prophet from the 'prophet' library
import pandas as pd
import matplotlib.pyplot as plt

# Load the original data and filter for California
data_path = '/content/NationalAndStatePregnancy_PublicUse.csv'
dataframe = pd.read_csv(data_path)

# Filter data for Texas
state_name = 'CA'
if state_name in dataframe['state'].values:
    california_data = dataframe[dataframe['state'] == state_name].copy()
    print(len(california_data))
    print(len(dataframe))
    # Ensure 'year' column is in datetime format
    california_data['year'] = pd.to_datetime(california_data['year'], format='%Y')

    # Prepare data for Prophet - rename columns to 'ds' and 'y' for compatibility
    forecast_data = california_data[['year', 'miscarriagestotal']].rename(columns={'year': 'ds', 'miscarriagestotal': 'y'})

    # Initialize and fit the Prophet model
    model = Prophet(yearly_seasonality=True)
    model.fit(forecast_data)

    # Generate future dates (up to 2030)
    future = model.make_future_dataframe(periods=11, freq='Y')  # 10 more years after 2020

    # Predict future values
    forecast = model.predict(future)

    # Filter for forecasted years 2021-2030
    forecasted_years = forecast[forecast['ds'].dt.year.isin([2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030])]

    # Plot the forecast
    model.plot(forecast)
    plt.title(f"miscarriagestotal Forecast for {state_name} (till 2030)")
    plt.xlabel("Year")
    plt.ylabel("miscarriagestotal Total")
    plt.show()

    # Display forecasted data for the selected years
    print(forecasted_years[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
else:
    print(f"State '{state_name}' not found in the dataset.")

!pip install prophet  # Install the 'prophet' library

from prophet import Prophet  # Import Prophet from the 'prophet' library
import pandas as pd
import matplotlib.pyplot as plt

# Load the original data and filter for California
data_path = '/content/NationalAndStatePregnancy_PublicUse.csv'
dataframe = pd.read_csv(data_path)

# Filter data for Texas
state_name = 'TX'
if state_name in dataframe['state'].values:
    texas_data = dataframe[dataframe['state'] == state_name].copy()
    print(len(texas_data))
    print(len(dataframe))
    # Ensure 'year' column is in datetime format
    texas_data['year'] = pd.to_datetime(texas_data['year'], format='%Y')

    # Prepare data for Prophet - rename columns to 'ds' and 'y' for compatibility
    forecast_data = texas_data[['year', 'miscarriagestotal']].rename(columns={'year': 'ds', 'miscarriagestotal': 'y'})

    # Initialize and fit the Prophet model
    model = Prophet(yearly_seasonality=True)
    model.fit(forecast_data)

    # Generate future dates (up to 2030)
    future = model.make_future_dataframe(periods=11, freq='Y')  # 10 more years after 2020

    # Predict future values
    forecast = model.predict(future)

    # Filter for forecasted years 2021-2030
    forecasted_years = forecast[forecast['ds'].dt.year.isin([2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030])]

    # Plot the forecast
    model.plot(forecast)
    plt.title(f"miscarriagestotal Forecast for {state_name} (till 2030)")
    plt.xlabel("Year")
    plt.ylabel("miscarriagestotal Total")
    plt.show()

    # Display forecasted data for the selected years
    print(forecasted_years[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
else:
    print(f"State '{state_name}' not found in the dataset.")

!pip install prophet  # Install the 'prophet' library

from prophet import Prophet  # Import Prophet from the 'prophet' library
import pandas as pd
import matplotlib.pyplot as plt

# Load the original data and filter for California
data_path = '/content/NationalAndStatePregnancy_PublicUse.csv'
dataframe = pd.read_csv(data_path)

# Filter data for Texas
state_name = 'CA'
if state_name in dataframe['state'].values:
    california_data = dataframe[dataframe['state'] == state_name].copy()
    print(len(california_data))
    print(len(dataframe))
    # Ensure 'year' column is in datetime format
    california_data['year'] = pd.to_datetime(california_data['year'], format='%Y')

    # Prepare data for Prophet - rename columns to 'ds' and 'y' for compatibility
    forecast_data = california_data[['year', 'pregnancyratetotal']].rename(columns={'year': 'ds', 'pregnancyratetotal': 'y'})

    # Initialize and fit the Prophet model
    model = Prophet(yearly_seasonality=True)
    model.fit(forecast_data)

    # Generate future dates (up to 2030)
    future = model.make_future_dataframe(periods=11, freq='Y')  # 10 more years after 2020

    # Predict future values
    forecast = model.predict(future)

    # Filter for forecasted years 2021-2030
    forecasted_years = forecast[forecast['ds'].dt.year.isin([2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030])]

    # Plot the forecast
    model.plot(forecast)
    plt.title(f"pregnancyratetotal Forecast for {state_name} (till 2030)")
    plt.xlabel("Year")
    plt.ylabel("pregnancyratetotal Total")
    plt.show()

    # Display forecasted data for the selected years
    print(forecasted_years[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
else:
    print(f"State '{state_name}' not found in the dataset.")

!pip install prophet  # Install the 'prophet' library

from prophet import Prophet  # Import Prophet from the 'prophet' library
import pandas as pd
import matplotlib.pyplot as plt

# Load the original data and filter for California
data_path = '/content/NationalAndStatePregnancy_PublicUse.csv'
dataframe = pd.read_csv(data_path)

# Filter data for Texas
state_name = 'TX'
if state_name in dataframe['state'].values:
    texas_data = dataframe[dataframe['state'] == state_name].copy()
    print(len(texas_data))
    print(len(dataframe))
    # Ensure 'year' column is in datetime format
    texas_data['year'] = pd.to_datetime(texas_data['year'], format='%Y')

    # Prepare data for Prophet - rename columns to 'ds' and 'y' for compatibility
    forecast_data = texas_data[['year', 'pregnancyratetotal']].rename(columns={'year': 'ds', 'pregnancyratetotal': 'y'})

    # Initialize and fit the Prophet model
    model = Prophet(yearly_seasonality=True)
    model.fit(forecast_data)

    # Generate future dates (up to 2030)
    future = model.make_future_dataframe(periods=11, freq='Y')  # 10 more years after 2020

    # Predict future values
    forecast = model.predict(future)

    # Filter for forecasted years 2021-2030
    forecasted_years = forecast[forecast['ds'].dt.year.isin([2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030])]

    # Plot the forecast
    model.plot(forecast)
    plt.title(f"pregnancyratetotal Forecast for {state_name} (till 2030)")
    plt.xlabel("Year")
    plt.ylabel("pregnancyratetotal Total")
    plt.show()

    # Display forecasted data for the selected years
    print(forecasted_years[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
else:
    print(f"State '{state_name}' not found in the dataset.")
