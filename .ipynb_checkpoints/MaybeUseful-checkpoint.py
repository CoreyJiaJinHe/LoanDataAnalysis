#Garbage
#New Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier

def generate_knn_pipeline(df):
    # Define categorical and numerical features

    ordinal_categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed']
    onehot_categorical_features = ['Property_Area']
    outlier_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    numerical_features = df.columns.difference(ordinal_categorical_features + onehot_categorical_features + ['Loan_Status'])

    o_idx=[df.columns.get_loc(item) for item in outlier_features]
    def outlier_transformer(o_idx):
        for column in outlier_features:
            #As the numeric values present in the dataset are large, we need to account for it with a large z-score threshold.
            #If the z-score threshold is too low, we risk removing half of the dataset as outliers. See ApplicantIncome.
            outlier_threshold = 3 
            removed_outliers =(((df[column]-df[column].mean()).abs()>=(df[column].std()*outlier_threshold)))
            z_score_no_outlier_df = df[~removed_outliers]
        return z_score_no_outlier_df

    df=outlier_transformer(o_idx)
    
    c1_idx = [df.columns.get_loc(item) for item in ordinal_categorical_features]
    c2_idx = [df.columns.get_loc(item) for item in onehot_categorical_features]
    n_idx = [df.columns.get_loc(item) for item in numerical_features]
    
    # Create transformers for numerical and categorical features


    outlier_transformer=Pipeline(steps=[('outliers',outlier_transformer(o_idx))])

    numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    
    ordinal_categorical_transformer = Pipeline(steps=[('ordinal', OrdinalEncoder(handle_unknown='error'))])
    
    onehot_categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    column_imputer = Pipeline(steps=[
        ('imputer0', KNNImputer())
    ])    
    # Apply transformers to features using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('outlier1',outlier_transformer,o_idx),
            ('cat1', ordinal_categorical_transformer, c1_idx),
            ('cat2', onehot_categorical_transformer, c2_idx),
            ('num', numerical_transformer, n_idx),
        ])

    missing_value_imputer = ColumnTransformer(
    transformers=[
        ('imputer', column_imputer, c1_idx + c2_idx + n_idx)
    ])

    
    # Define the KNN model
    knn_model = KNeighborsClassifier(n_neighbors=3)


    
    # Create the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('imputer',missing_value_imputer),
        ('classifier', knn_model)
    ])

    return pipeline
df = load_data('data/loan.csv')

pipeline = generate_knn_pipeline(df)


from sklearn import set_config
set_config(display="diagram")
pipeline



# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Get predictions from test data
predictions = pipeline.predict(X_test)

# Get evaluation metrics
#cm = confusion_matrix(y_test,predictions)
#print('Confusion Matrix:\n',cm, '\n')
#print('Accuracy:', accuracy_score(y_test,predictions))
#print("Overall Precision:", precision_score(y_test,predictions))
#print("Overall Recall:", recall_score(y_test,predictions))


#New Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

df = load_data('data/loan.csv')

# Encode the target variable using LabelEncoder
label_encoder = LabelEncoder()
df['Loan_Status'] = label_encoder.fit_transform(df['Loan_Status'])

X_train, X_test, y_train, y_test = separate_features_from_label_and_split_the_data(df, features, label, test_size=0.3, random_state=0)

# Define categorical and numerical features

c2_idx = [df.columns.get_loc(item) for item in onehot_categorical_features]
n_idx = [df.columns.get_loc(item) for item in numerical_features]

# Create transformers for numerical and categorical features




# Apply transformers to features using ColumnTransformer
feature_transformer = ColumnTransformer(
    transformers=[
        
        ('cat1', ordinal_categorical_transformer, c1_idx),
        ('cat2', onehot_categorical_transformer, c2_idx),
        ('num', numerical_transformer, n_idx),
    ])

missing_value_imputer = ColumnTransformer(
    transformers=[
        ('imputer', column_imputer, o_idx+ c1_idx + c2_idx + n_idx)
    ])

# Define the KNN model
knn_model = KNeighborsClassifier(n_neighbors=3)  # You can adjust the number of neighbors

# Create the pipeline
pipeline = Pipeline(steps=[('transformer',feature_transformer),('imputer',missing_value_imputer),('classifier',knn_model)])

from sklearn import set_config
set_config(display="diagram")
pipeline



# Fit the pipeline on the training data
#pipeline.fit(X_train, y_train)

# Get predictions from test data
#predictions = pipeline.predict(X_test)

# Get evaluation metrics
#cm = confusion_matrix(y_test,predictions)
#print('Confusion Matrix:\n',cm, '\n')
#print('Accuracy:', accuracy_score(y_test,predictions))
#print("Overall Precision:", precision_score(y_test,predictions))
#print("Overall Recall:", recall_score(y_test,predictions))

