import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

class SmartGridDataLoader:
    def __init__(self, path):
        self.path = path
        self.df = None

    def load(self):
        self.df = pd.read_csv(self.path)
        print("✅ Data Loaded.")
        return self.df

    def inspect(self):
        print("\n📋 Head:\n", self.df.head())
        print("\n🔍 Missing Values:\n", self.df.isnull().sum())
        print("\n📊 Summary:\n", self.df.describe())
        print("\n🔢 Column Count:\n", self.df.count())
        print("\nℹ️ Info:")
        print(self.df.info())

class SmartGridVisualizer:
    def __init__(self, df):
        self.df = df

    def plot_distributions(self):
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        sns.histplot(self.df['tau1'], bins=30, kde=True, ax=axes[0, 0], color='skyblue')
        sns.histplot(self.df['p1'], bins=30, kde=True, ax=axes[0, 1], color='olive')
        sns.histplot(self.df['g1'], bins=30, kde=True, ax=axes[1, 0], color='gold')
        sns.histplot(self.df['stab'], bins=30, kde=True, ax=axes[1, 1], color='teal')
        axes[0, 0].set_title('Distribution of tau1')
        axes[0, 1].set_title('Distribution of p1')
        axes[1, 0].set_title('Distribution of g1')
        axes[1, 1].set_title('Distribution of stab')
        plt.tight_layout()
        plt.show()

    def correlation_heatmap(self):
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        correlation_matrix = self.df[numeric_cols].corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Correlation Matrix")
        plt.show()
        print("Correlation with 'stab':\n", correlation_matrix['stab'].sort_values(ascending=False))

    def boxplots_by_class(self):
        plt.figure(figsize=(16, 8))
        for i, column in enumerate(self.df.columns[:-2]):
            plt.subplot(3, 4, i + 1)
            sns.boxplot(x='stabf', y=column, data=self.df)
            plt.title(column)
        plt.tight_layout()
        plt.show()

class SmartGridPreprocessor:
    def __init__(self, df):
        self.df = df
        self.le = LabelEncoder()

    def encode_and_split(self):
        self.df['stabf_encoded'] = self.le.fit_transform(self.df['stabf'])
        X = self.df.drop(['stabf', 'stabf_encoded', 'stab'], axis=1)
        y = self.df['stabf_encoded']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, self.le

class SmartGridModel:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'SVM': SVC(kernel='linear', random_state=42),
            'XGBoost': XGBClassifier(random_state=42)
        }
        self.results = {}

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, label_encoder):
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            report = classification_report(y_test, preds, target_names=label_encoder.classes_)
            self.results[name] = {
                'accuracy': acc,
                'report': report,
                'model': model
            }

    def print_results(self):
        for name, res in self.results.items():
            print(f"\n🔍 {name} Classifier Report:")
            print(f"Accuracy: {res['accuracy']:.4f}")
            print("Classification Report:\n", res['report'])

    def plot_feature_importance(self, feature_names):
        rf_model = self.results['Random Forest']['model']
        importances = rf_model.feature_importances_
        series = pd.Series(importances, index=feature_names)
        plt.figure(figsize=(10, 8))
        series.sort_values().plot(kind='barh', color='skyblue')
        plt.title('Feature Importances (Random Forest)')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.show()

class SmartGridRunner:
    def __init__(self, csv_path):
        self.path = csv_path

    def run(self):
        # Phase 1-2: Load + Inspect
        loader = SmartGridDataLoader(self.path)
        df = loader.load()
        loader.inspect()

        # Phase 3-5: Visualizations
        visualizer = SmartGridVisualizer(df)
        visualizer.plot_distributions()
        visualizer.correlation_heatmap()
        visualizer.boxplots_by_class()

        # Phase 6: Preprocessing
        preprocessor = SmartGridPreprocessor(df)
        X_train, X_test, y_train, y_test, le = preprocessor.encode_and_split()

        # Phase 7-10: Train + Evaluate
        model_runner = SmartGridModel()
        model_runner.train_and_evaluate(X_train, X_test, y_train, y_test, le)
        model_runner.print_results()
        model_runner.plot_feature_importance(X_train.columns)

# 🔁 Run the pipeline
if __name__ == "__main__":
    path = r'C:\Users\shoaib.ahmad\Desktop\power system\Stability-by-Design-Machine-Learning-Models-for-Smart-Grid-Predictions-main\smart_grid_stability.csv'
    runner = SmartGridRunner(path)
    runner.run()
