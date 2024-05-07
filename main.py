import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def feature_extraction_on_file(dataframeORG, train_idx, val_idx):
    # dataframe = pd.read_pickle(r'training_set.pkl')
    dataframe = dataframeORG.copy()

    def normalize_features(dataframe):
        """
        Normalize all numeric features in a DataFrame using the formula: (value - min) / (max - min).

        """
        dataframe_normalized = dataframe.copy()

        for column in dataframe.select_dtypes(include='number').columns:
            min_value = dataframe_normalized[column].iloc[train_idx].min()
            max_value = dataframe_normalized[column].iloc[train_idx].max()

            if max_value - min_value != 0:
                dataframe_normalized[column] = (dataframe_normalized[column] - min_value) / (max_value - min_value)

        for column in dataframe.columns:
            unique_values = dataframe[column].unique()

            if len(unique_values) == 2 and pd.api.types.is_bool_dtype(unique_values):
                dataframe_normalized[column] = dataframe_normalized[column].astype(int)

        return dataframe_normalized

    dataframe = normalize_features(dataframe)

    #normalize the categorial values - one hot

    #category

    def create_categories_binary(dataframe, categorical_features):
        """
        Create new binary columns for each category in the specified categorical features.

        """

        dataframe_binary = dataframe.copy()

        for feature in categorical_features:
            unique_categories = dataframe_binary[feature].unique()

            for category in unique_categories:
                binary_column_name = '{}_{}_binary'.format(feature, category)
                dataframe_binary[binary_column_name] = (dataframe_binary[feature] == category).astype(int)

        return dataframe_binary

    categorical_features = ['type']
    dataframe = create_categories_binary(dataframe, categorical_features)
    return dataframe


df = pd.read_excel('data2.xlsx', engine='openpyxl')
X_train = df.drop(['Has_Ordered'], axis=1, errors='ignore')
y_train = df['Has_Ordered']
kfold = KFold(n_splits=10, shuffle=True, random_state=123)
max_depth_list = np.arange(1, 4, 1)
res = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):  # for each train test split
    # Apply feature extraction on X_train and X_val for this fold
    X_train_normal = feature_extraction_on_file(X_train, train_idx, val_idx)
    X_train_normal.drop(['type'], axis=1, errors='ignore', inplace=True)
    X_train_fold = X_train_normal.iloc[train_idx]
    X_val_fold = X_train_normal.iloc[val_idx]
    # X_train_fold = X_train_fold.drop(['Has_Ordered'], axis=1, errors='ignore')
    # X_val_fold = X_val_fold.drop(['Has_Ordered'], axis=1, errors='ignore')
    X_val_fold = X_val_fold[X_train_fold.columns]
    for max_depth in max_depth_list:  # for each depth value we are checking

        # define model
        model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=42)

        # train it using the training indexes
        model.fit(X_train_fold, y_train.iloc[train_idx])

        # test it and get accuracy using the validation indexes
        # acc = roc_auc_score(y_train.iloc[val_idx], model.predict(X_val_fold))
        acc = roc_auc_score(y_train.iloc[train_idx], model.predict(X_train_fold))
        # save results
        res.append({'fold': fold + 1,
                    'max_depth': max_depth,
                    'acc': acc})

res = pd.DataFrame(res)

average_res=res.groupby(['max_depth'])['acc'].mean().reset_index().sort_values('acc',ascending=False)

print(average_res)

# best_max_depth = average_res.loc[2,'max_depth']
best_max_depth = average_res.loc[average_res['acc'].idxmax(), 'max_depth']
print(best_max_depth)





#### tree plot with best levels ######
##########gini###################3
df = pd.read_excel('data2.xlsx', engine='openpyxl')
X = df.drop(['Has_Ordered'], axis=1)
y = df['Has_Ordered']
# split the dataset into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

train_idx = X_train.index
test_idx = X_test.index
df_normal = feature_extraction_on_file(df, train_idx.union(test_idx), test_idx)
df_normal.drop(['type'], axis=1, errors='ignore', inplace=True)
X_train= df_normal.iloc[train_idx]
X_test = df_normal.iloc[test_idx]
X_train = X_train.drop(['Has_Ordered'], axis=1, errors='ignore')
X_test = X_test.drop(['Has_Ordered'], axis=1, errors='ignore')
X_test = X_test[X_train.columns]
model = DecisionTreeClassifier(criterion='entropy', max_depth=best_max_depth, random_state=42)
model.fit(X_train, y_train)
#plot tree to see decision funtoin
plt.figure(figsize=(12,10))
plot_tree(model, filled=True, class_names=True, feature_names=X_train.columns.to_list())
plt.show()