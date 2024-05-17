import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
from collections import Counter

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
        self.feature_importances_ = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.feature_importances_ = np.zeros(X.shape[1])
        self.root = self._grow_tree(X, y)
        self.feature_importances_ /= self.feature_importances_.sum()  # Normalize importances

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feature is not None:
            left_idxs, right_idxs = self._split(X.iloc[:, best_feature], best_thresh)
            left = self._grow_tree(X.iloc[left_idxs, :], y.iloc[left_idxs], depth + 1)
            right = self._grow_tree(X.iloc[right_idxs, :], y.iloc[right_idxs], depth + 1)
            self.feature_importances_[best_feature] += self._information_gain(y, X.iloc[:, best_feature], best_thresh)
            return Node(best_feature, best_thresh, left, right)

        leaf_value = self._most_common_label(y)
        return Node(value=leaf_value)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X.iloc[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        left_labels = y.iloc[left_idxs]
        right_labels = y.iloc[right_idxs]

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(left_labels), self._entropy(right_labels)
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        if isinstance(split_thresh, str):
            left_idxs = np.where(X_column == split_thresh)[0]
            right_idxs = np.where(X_column != split_thresh)[0]
        else:
            left_idxs = np.argwhere(X_column <= split_thresh).flatten()
            right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        y_values = y.values if isinstance(y, pd.Series) else y
        hist = np.bincount(y_values)
        ps = hist / len(y_values)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X.to_numpy()])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if node.feature >= len(x):
            return node.value

        if isinstance(node.threshold, str):
            if x[node.feature] == node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)
        else:
            if isinstance(x[node.feature], str):
                if x[node.feature] == node.threshold:
                    return self._traverse_tree(x, node.left)
                else:
                    return self._traverse_tree(x, node.right)
            else:
                if x[node.feature] <= node.threshold:
                    return self._traverse_tree(x, node.left)
                else:
                    return self._traverse_tree(x, node.right)

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_feature=None):
        self.n_trees = n_trees
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.n_features=n_feature
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                            min_samples_split=self.min_samples_split,
                            n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X.iloc[idxs], y.iloc[idxs]


    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions

def model_predictions(model, X_train, X_test, y_train, y_test, df, model_name):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    df, model_confusion = performance_metrics_recorder(predictions, y_test, df, model_name)
    return model, df, model_confusion

def performance_metrics_recorder(predictions, y_test, df, model_name):
    model_accuracy_score = accuracy_score(y_test, predictions)
    model_precision_score = precision_score(y_test, predictions)
    model_recall_score = recall_score(y_test, predictions)
    model_f1_score = f1_score(y_test, predictions)
    model_roc_auc = roc_auc_score(y_test, predictions)
    model_confusion = confusion_matrix(y_test, predictions)

    df.loc[len(df)] = [model_name, model_accuracy_score, model_precision_score, model_recall_score, model_f1_score, model_roc_auc]
    return df, model_confusion

def feature_importance_visualizer(X, model, label, color=None, grid=None):
    if isinstance(model, DecisionTree):
        feature_importance = model.feature_importances_
    elif isinstance(model, RandomForest):
        feature_importance = np.mean([tree.feature_importances_ for tree in model.trees], axis=0)
    else:
        raise ValueError("Unsupported model type. Supported types are DecisionTree and RandomForest.")

    features_list = X.columns.values
    sorted_idx = np.argsort(feature_importance)

    plt.figure()
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center', color=color)
    plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
    plt.xlabel('Importance')
    plt.title("Feature Importances According to " + label + " Classifier")
    if grid:
        plt.grid(alpha=0.5)
    plt.show()
    
# Reading the raw CSV data file
df = pd.read_csv("Dataset/mushrooms.csv")

# Creating a copy of the original dataframe
df2 = df.copy()

# Encoding categorical variables
label_encoder = LabelEncoder()
for col in df2.columns:
    df2[col] = label_encoder.fit_transform(df2[col])

# Splitting dataset
X = df2.drop(['class'], axis=1)
Y = df2['class']

# Creating train/test split using 80% data for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Defining an empty df to record metrics from each model and stage of analysis
column_names = ["method", "accuracy", "precision", "recall", "f1", "roc_auc"]
performance_metrics = pd.DataFrame(columns=column_names)

label = "Decision Tree"
dt = DecisionTree()
dt, performance_metrics, dt_confusion = model_predictions(dt, X_train, X_test, y_train, y_test, performance_metrics, label)
feature_importance_visualizer(X, dt, label, color="blueviolet")

# Random Forest
label = "Random Forest"
rf = RandomForest(n_trees=20)
rf, performance_metrics, rf_confusion = model_predictions(rf, X_train, X_test, y_train, y_test, performance_metrics, label)
feature_importance_visualizer(X, rf, label, color="blueviolet")

print(performance_metrics)
print(dt_confusion)