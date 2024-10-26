import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend to avoid GUI issues
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed dataset
preprocessed = 'datasets/preprocessed.csv'  # Adjust this path if necessary

# Load the data from the CSV file
data = pd.read_csv(preprocessed)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Check the number of existing data points
existing_points = len(data)
print(f"Existing data points: {existing_points}")

# Calculate how many more points are needed to reach 150
num_additional_points = max(0, 150 - existing_points)

# Simulate additional data points if necessary
if num_additional_points > 0:
    # Create new dates starting from the last date in the existing dataset
    last_date = data['Date'].max()
    new_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_additional_points)

    # Simulate new exchange rates based on the existing ones
    last_exchange_rate = data['Exchange rate'].iloc[-1]
    # Generate random exchange rates around the last known value with some noise
    new_exchange_rates = np.random.normal(loc=last_exchange_rate, scale=0.01, size=num_additional_points)

    # Create a new DataFrame for the simulated data
    new_data = pd.DataFrame({
        'Date': new_dates,
        'Country': ['Australia'] * num_additional_points,
        'Exchange rate': new_exchange_rates
    })

    # Combine original and new data
    combined_data = pd.concat([data, new_data], ignore_index=True)
else:
    combined_data = data

# Set the style for seaborn
sns.set(style="whitegrid")

# Create and save a line plot for exchange rates over time with moving average
combined_data['Moving Average'] = combined_data['Exchange rate'].rolling(window=30).mean()  # 30-day moving average

plt.figure(figsize=(12, 6))
plt.plot(combined_data['Date'], combined_data['Exchange rate'], label='Exchange Rate', color='blue', alpha=0.5)
plt.plot(combined_data['Date'], combined_data['Moving Average'], label='30-Day Moving Average', color='orange', linestyle='--')
plt.title('Exchange Rate Over Time with Moving Average')
plt.xlabel('Date')
plt.ylabel('Exchange Rate')
plt.xticks(rotation=45)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('exchange_rate_over_time_with_moving_average.png')  # Save the figure
plt.close()

# Create and save a histogram of exchange rates with fewer bins for clarity
plt.figure(figsize=(12, 6))
sns.histplot(combined_data['Exchange rate'], bins=30, kde=True)  # Adjust number of bins based on data distribution
plt.title('Distribution of Exchange Rates')
plt.xlabel('Exchange Rate')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('exchange_rate_distribution.png')  # Save the figure
plt.close()

# Create and save a boxplot for exchange rates to visualize spread and outliers
plt.figure(figsize=(12, 6))
sns.boxplot(x=combined_data['Exchange rate'])
plt.title('Boxplot of Exchange Rates')
plt.xlabel('Exchange Rate')
plt.tight_layout()
plt.savefig('exchange_rate_boxplot.png')  # Save the figure
plt.close()

# Create and save a scatter plot of Date vs Exchange Rate with annotations for key points if needed
plt.figure(figsize=(12, 6))
sns.scatterplot(x=combined_data['Date'], y=combined_data['Exchange rate'], color='red', alpha=0.5)
for i in range(len(combined_data)):
    if i % 50 == 0:  # Annotate every 50th point for clarity (adjust as needed)
        plt.text(combined_data['Date'][i], combined_data['Exchange rate'][i], f"{combined_data['Exchange rate'][i]:.2f}", fontsize=8)

plt.title('Scatter Plot of Exchange Rates Over Time')
plt.xlabel('Date')
plt.ylabel('Exchange Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('exchange_rate_scatter_plot.png')  # Save the figure
plt.close()