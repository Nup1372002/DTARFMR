import numpy as np
import pandas as pd
from sklearn import tree
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
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, entropy=None, value_counts=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.entropy = entropy
        self.value_counts = value_counts

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
            return Node(value=leaf_value, entropy=self._entropy(y), value_counts=dict(Counter(y)))

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feature is not None:
            left_idxs, right_idxs = self._split(X.iloc[:, best_feature], best_thresh)
            left = self._grow_tree(X.iloc[left_idxs, :], y.iloc[left_idxs], depth + 1)
            right = self._grow_tree(X.iloc[right_idxs, :], y.iloc[right_idxs], depth + 1)
            self.feature_importances_[best_feature] += self._information_gain(y, X.iloc[:, best_feature], best_thresh)
            return Node(best_feature, best_thresh, left, right, entropy=self._entropy(y),
                        value_counts=dict(Counter(y)))

        leaf_value = self._most_common_label(y)
        return Node(value=leaf_value, entropy=self._entropy(y), value_counts=dict(Counter(y)))

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
    def plot_tree(self):
        plt.figure(figsize=(20, 10))
        self._plot_tree_recursive(self.root, x_offset=0.5, y_offset=1, indent=0, level=0, width=1.0)
        plt.show()

    def _plot_tree_recursive(self, node, x_offset, y_offset, indent, level, width):
        if node is None:
            return

        if node.is_leaf_node():
            plt.text(x_offset, y_offset, f"Class {node.value}\nEntropy: {node.entropy:.2f}\nValue: {node.value_counts}", ha='center', va='center', bbox=dict(facecolor='lightgray', alpha=0.5))
            return

        plt.text(x_offset, y_offset, f"X[{node.feature}] <= {node.threshold}\nEntropy: {node.entropy:.2f}\nValue: {node.value_counts}", ha='center', va='center', bbox=dict(facecolor='lightgray', alpha=0.5))
        if node.left is not None:
            plt.plot([x_offset, x_offset - 0.1 * width], [y_offset - 0.1, y_offset - 1], '-k')
            new_x_offset = x_offset - 0.1 * width
            self._plot_tree_recursive(node.left, new_x_offset, y_offset - 1, indent + 1, level + 1, width * 0.8)  # Adjust width here
        if node.right is not None:
            plt.plot([x_offset, x_offset + 0.1 * width], [y_offset - 0.1, y_offset - 1], '-k')
            new_x_offset = x_offset + 0.1 * width
            self._plot_tree_recursive(node.right, new_x_offset, y_offset - 1, indent + 1, level + 1, width * 0.8)  # Adjust width here

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
    features_list = X.columns.values
    feature_importance = model.feature_importances_
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

# Decision Tree
label = "Decision Tree"
dt = DecisionTree()
dt, performance_metrics, dt_confusion = model_predictions(dt, X_train, X_test, y_train, y_test, performance_metrics, label)

# Visualize feature importance
feature_importance_visualizer(X, dt, label, color="blueviolet")

print(performance_metrics)
print(dt_confusion)
dt.plot_tree()