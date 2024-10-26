# Model for USA/EUR Foreign Exchange Rate Prediction
Create a model that can predict USA/EUR foreign exchange rate depending on the country and based on history data.

## Dataset Description
The dataset contains data from multiple countries with their conversion rate starting from 1971 to 2018. It is from U.S. Dollars to the respective countries' currencies.

## Summary of Findings
(summary_of_findings) - [text form] This will be a brief summary of the findings of the project.
The Linear Regression Model was effective in predicting the foreign exchange rates even outside the range of the dataset dates. For example, it was able to predict Malaysia's currency exchange in 2025/05/24 to be 3.93

## Data Preprocessing
The rows with empty foreign exchange rates were dropped from the dataset.


## Exploratory Data Analysis
### Visualization
(visualization) - [image form] This will be a visualization of the data that you have chosen to represent in the project; each visualization must have a corresponding interpretation. All visualizations must be placed here.

## Model Development
Four (4) algorithms were used: linear regression model, long short-term memory, support vector machine model, and random forest. Each model had their own specifications and the training and testing of data. The training and testing was split to 80/20 respectively. Later on, the models were saved as .pkl for further use

## Model Evaluation
Most models used the RSE and MSE for the model evaluation as well as the classification model used for finding accuracy and precision

## Conclusion
The linear regression model is the most effective for few column datasets. It can predict values out of its dataset for future dates, especially for the 