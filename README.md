# Customer Churn Prediction for a Telecom Company

(<i>Please go [here](https://github.com/jojouna/telecom_customer_churn_predict/blob/main/customerChurnPrediction_v2.ipynb) to see the whole work process</i>)

## Assignment
You are provided with a sample dataset of a telecom company’s customers and it's expected to done the following tasks:

- Perform exploratory analysis and extract insights from the dataset.
- Split the dataset into train/test sets and explain your reasoning.
- Build a predictive model to predict which customers are going to churn and discuss the reason why you choose a particular algorithm.
- Establish metrics to evaluate model performance.
- Discuss the potential issues with deploying the model into production.


## Answers
### 1. Insights from the dataset
- There are total 3333 rows with 19 features
- Most of the customers don't use voice mail messages
- Mean of the total charge was the highest during the day (30.562307) compared to evening (17.083540) and night (9.039325).
- The distribution for total calls made by the customer have similar distribution across day, evening and night.
- International call was the lowest compared to the day, evening and night calls.
- Customer service calls are skewed to the right with most of the calls made were only 1-2 times.
- This is an imbalanced dataset. The number of churn false is 2850 whereas true is only 483. We will stratify when spliting it into train and test sets.

### 2. Train/test set split reasoning
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

- Set the test size to 20% since that is the largely accepted standard.
- Stratify by y since the dataset is imbalanced so that we don't have either train or test set to be imbalanaced.

### 3. Build a predictive model to predict which customers are going to churn and discuss the reason why you choose a particular algorithm.
I have choose Random Forest Classifier, Naive Bayes, Logistics Regression, Decision Tree Classifier, Support Vector Machine and K-Neighbors classifier. The reason for choosing these is because,

- they are well-established supervised learning model,
- able to handle both numerical and categorical variables,
- diverse in terms of learning strategies (e.g., tree-based, distance-based, probabilistic, margin-based), allowing for a broad comparison.

### 4. Establish metrics to evaluate model performance

```
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```
```
              precision    recall  f1-score   support

       False       0.95      0.99      0.97       570
        True       0.92      0.69      0.79        97

    accuracy                           0.95       667
   macro avg       0.93      0.84      0.88       667
weighted avg       0.94      0.95      0.94       667
```

### 5. Discuss the potential issues with deploying the model into production
There could be several deployment issues with the model.

1. Data Drift
Over time, the behaviour of the customers may change, so we have to keep monitor the customer data and update the dataset to the very recent one.

2. Model Staleness
The model is trained on historical data. As the business or customer base evolves, older patterns may no longer apply. To solve this we should establish a regular retraining schedule with updated data.

3. Data Privacy and Compliance
Customer data could often be sensitive. We should ensure the model and data pipeline comply with GDPR, CCPA and company policies.
