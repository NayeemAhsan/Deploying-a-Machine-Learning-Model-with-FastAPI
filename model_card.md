
# Model Card

  
## Model Details
- This classifier trained to predict whether an employer's  income exceeds $50K/year.
- **Random Forest** and **Logistic Regression** were used to assess at EDA. **Random Forest** will be considered for model training and evaluation since it shows better results.


## Intended Use
- Intended to be used to determine what features impacts the income of a person.
- Intended to determine underprivileged employers.
- Not suitable for modern dates since the data is quite old.
  
## Factors
 - Evaluate on features that may be underprivileged such as gender, race, etc.
 
## Training Data
- Census Income [Dataset](https://archive.ics.uci.edu/ml/datasets/census+income) from UCI
- Categorical data:
  - Handled missing values by imputing the data using `SimpleImputer` with the most frequent value
  - Encoded the categories using `OneHotEncoder` and setting the value of `handle_unknown='ignore'`
- Numerical data:
  - Handled missing values by imputing the data using `SimpleImputer` with 'constant' strategy

## Evaluation Data
- Splitting the train data using sklearn `train_test_split` with a fixed `random_state=42` and stratified on `salary label`.
  
## Metrics
- Evaluation metrics includes **Precision**, **Recall** and **F1 beta score**.
- These 3 metrics can be calculated from the confusion matrix for binary classification which are more suitable for imbalanced problems.
- Precision: Ratio between correct predictions and the total predictions
- Recall: Ratio of the correct predictions and the total number of correct items in the set
- F1 Beta: Harmoinc mean between Precision and Recall to show the balance between them.

## Ethical Considerations
- Data is open sourced on UCI machine learning repository for educational purposes.

## Caveats and Recommendations
- The data was collected in 1996 which does not reflect insights from the modern world.
- Features with minor categories should be focused more when collecting extra data.

## Quantitative Analyses
All results shown are calculated for class 1 (>50K) using sklearn metrics
|				    |Train |Validation|Test  |
|-----------|------|----------|------
|Precision	|1.000 |0.729     |0.689 |
|Recall     |1.000 |0.623     |0.639 |
|F1 Beta    |0.999 |0.672     |0.747 |

