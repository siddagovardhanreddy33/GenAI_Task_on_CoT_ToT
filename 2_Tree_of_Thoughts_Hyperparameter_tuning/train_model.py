
import pandas as pd
import time

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


def train_and_evaluate():
    X, y = load_iris(return_X_y=True)
    configs = []
    config_id = 1
    
    for n_estimators in [50, 100, 200]:
        for max_depth in [3, 5, 10]:
            start_time = time.time()
            model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,min_samples_split=2,random_state=42)
            cv_scores = cross_val_score( model,X,y,cv=5,scoring="accuracy")
            cv_accuracy = cv_scores.mean()
            model.fit(X, y)

            train_pred = model.predict(X)
            train_accuracy = accuracy_score(y, train_pred)
            training_time = time.time() - start_time

            gap = train_accuracy - cv_accuracy

            configs.append({
                "Config": f"C{config_id}",
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": 2,
                "CV_Accuracy": round(cv_accuracy, 4),
                "Train_Accuracy": round(train_accuracy, 4),
                "Gap": round(gap, 4),
                "Training_Time": round(training_time, 4)
            })
            config_id += 1

    df = pd.DataFrame(configs)

    df.to_csv("outputs/hyperparameter_results.csv",index=False)
    print(df)
    return df


if __name__ == "__main__":
    train_and_evaluate()