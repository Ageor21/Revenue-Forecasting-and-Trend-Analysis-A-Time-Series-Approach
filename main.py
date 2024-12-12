# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error
from scipy.signal import periodogram
from nbconvert import HTMLExporter
import nbformat

# Load dataset
file_path = 'medical_clean.csv'
data = pd.read_csv(file_path)

# Add differenced column for stationarity
data['Differenced_Revenue'] = data['Revenue'].diff()

# Drop NaN values
data_cleaned = data.dropna().reset_index(drop=True)

# Splitting the dataset into training and testing sets (80/20 split)
split_index = int(len(data_cleaned) * 0.8)
training_data = data_cleaned.iloc[:split_index]
test_data = data_cleaned.iloc[split_index:]

# Plotting trends in original and differenced data
plt.figure(figsize=(12, 6))
plt.plot(data_cleaned['Day'], data_cleaned['Revenue'], label='Original Revenue', color='blue', linewidth=1)
plt.plot(data_cleaned['Day'], data_cleaned['Differenced_Revenue'], label='Differenced Revenue', color='orange', linewidth=1)
plt.title('Trends in the Time Series')
plt.xlabel('Day')
plt.ylabel('Revenue (in million dollars)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Plotting Autocorrelation Function (ACF)
plt.figure(figsize=(12, 6))
plot_acf(data_cleaned['Differenced_Revenue'], lags=50)
plt.title('Autocorrelation Function (ACF)')
plt.show()

# Spectral Density
freqs, power = periodogram(data_cleaned['Revenue'], scaling='density')
plt.figure(figsize=(12, 6))
plt.semilogy(freqs, power, label='Spectral Density')
plt.title('Spectral Density of Revenue Time Series')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Seasonal Decomposition
decompose_result = seasonal_decompose(data_cleaned['Revenue'], model='additive', period=365)
decompose_result.plot()
plt.show()

# Residual Analysis
residuals = decompose_result.resid.dropna()
plt.figure(figsize=(12, 6))
plt.plot(residuals, label='Residuals', color='purple', linewidth=1)
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.title('Residuals of Decomposed Series')
plt.legend()
plt.show()

# Fitting ARIMA model
arima_model = ARIMA(training_data['Revenue'], order=(1, 1, 1))
arima_result = arima_model.fit()
print(arima_result.summary())

# Forecasting the next 30 days
forecast_steps = 30
forecast_result = arima_result.get_forecast(steps=forecast_steps)
forecast_mean = forecast_result.predicted_mean
forecast_ci = forecast_result.conf_int()

# Evaluate the forecast with Mean Absolute Error (MAE)
aligned_actual_test_values = test_data['Revenue'].values[:forecast_steps]
aligned_forecast_mean = forecast_mean[:len(aligned_actual_test_values)]
mae = mean_absolute_error(aligned_actual_test_values, aligned_forecast_mean)

# Forecast Visualization
plt.figure(figsize=(12, 6))
plt.plot(data_cleaned['Day'], data_cleaned['Revenue'], label='Observed Revenue', color='blue')
plt.plot(test_data['Day'][:forecast_steps], aligned_actual_test_values, label='Actual Test Set', color='green')
plt.plot(test_data['Day'][:forecast_steps], aligned_forecast_mean, label='Forecast', color='orange')
plt.fill_between(test_data['Day'][:forecast_steps],
                 forecast_ci.iloc[:forecast_steps, 0],
                 forecast_ci.iloc[:forecast_steps, 1], color='orange', alpha=0.3, label='Confidence Interval')
plt.legend()
plt.title('Forecast vs Test Set')
plt.xlabel('Day')
plt.ylabel('Revenue (in million dollars)')
plt.grid(alpha=0.3)
plt.show()

# Display MAE
print(f"Mean Absolute Error (MAE): {mae}")

# Create a Jupyter Notebook for the analysis
notebook_content = nbformat.v4.new_notebook()
notebook_content.cells.append(nbformat.v4.new_markdown_cell("# Time Series Analysis Report"))

# Save and convert to HTML
notebook_path = 'Time_Series_Analysis_Report.ipynb'
with open(notebook_path, 'w') as notebook_file:
    nbformat.write(notebook_content, notebook_file)
html_exporter = HTMLExporter()
html_exporter.exclude_input = True
(html_body, _) = html_exporter.from_filename(notebook_path)
html_path = 'Time_Series_Analysis_Report.html'
with open(html_path, 'w') as html_file:
    html_file.write(html_body)