import warnings
# import argparse
import logging
import pandas as pd
import numpy as np

# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.svm import SVC
from sklearn import preprocessing


import mlflow
import mlflow.sklearn
from pathlib import Path
import os

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# get arguments from command
# parser = argparse.ArgumentParser()
# parser.add_argument("--alpha", type=float, required=False, default=0.7)
# parser.add_argument("--l1_ratio", type=float, required=False, default=0.7)
# args = parser.parse_args()

# evaluation function
def eval_metrics(actual, pred):
    # rmse = np.sqrt(mean_squared_error(actual, pred))
    # mae = mean_absolute_error(actual, pred)
    # r2 = r2_score(actual, pred)
    accuracy = accuracy_score(actual, pred)
    recall = recall_score(actual, pred, average="macro")
    precision = precision_score(actual, pred, average="macro")
    f1 = f1_score(actual, pred, average="macro")
    return accuracy,precision, recall, f1


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    data = pd.read_csv("data\Crop_recommendation.csv")
    #os.mkdir("data/")
    # data.to_csv("data/red-wine-quality.csv", index=False)

    # melakukan encoder
    label_list = data['label']
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit_transform(data['label'])
    data['label'] = label_encoder.transform(data['label'])

    # pembagian dataset untuk input dan target
    X = data.drop(['label'], axis=1).values
    y = data['label'].values

    #pembagian data menggunakan train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Split the data into training and test sets. (0.75, 0.25) split.
    # train, test = train_test_split(data)
    # train.to_csv("data/train.csv")
    # test.to_csv("data/test.csv")
    # # The predicted column is "quality" which is a scalar from [3, 9]
    # train_x = train.drop(["quality"], axis=1)
    # test_x = test.drop(["quality"], axis=1)
    # train_y = train[["quality"]]
    # test_y = test[["quality"]]

    # alpha = args.alpha
    # l1_ratio = args.l1_ratio

    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    print("The set tracking uri is ", mlflow.get_tracking_uri())

###########First Experiment Elastic Net #############
    print("First Experiment SVC")
    exp = mlflow.set_experiment(experiment_name="exp_cls_SVC")

    print("Name: {}".format(exp.name))
    print("Experiment_id: {}".format(exp.experiment_id))
   # print("Artifact Location: {}".format(exp.artifact_location))
   # print("Tags: {}".format(exp.tags))
   # print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
   # print("Creation timestamp: {}".format(exp.creation_time))

    mlflow.start_run(run_name="run1.1")
    tags = {
        "engineering": "ML platform",
        "release.candidate":"RC1",
        "release.version": "1.0"
    }

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    svc = SVC(C=100)
    svc.fit(x_train, y_train)

    predicted = svc.predict(x_test)
    accuracy, precision, recall, f1_score = eval_metrics(y_test, predicted)

    print("SVC model")
    print("  Accuracy: %s" % accuracy)
    print("  Precision: %s" % precision)
    print("  Recall: %s" % recall)
    print("  F1-Score: %s" % f1_score)
    # lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    # lr.fit(train_x, train_y)

    # predicted_qualities = lr.predict(test_x)
    # (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    # print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    # print("  RMSE: %s" % rmse)
    # print("  MAE: %s" % mae)
    # print("  R2: %s" % r2)
    #log parameters

    # mlflow.autolog(
    #     log_input_examples=True
    # )
    params ={
        "C": 100,
    }
    mlflow.log_params(params)
    # #log metrics
    metrics = {
        "accuracy":accuracy,
        "precision":precision,
        "recall":recall,
        "f1-score": f1_score,
    }
    mlflow.log_metrics(metrics)
    #log model
    mlflow.sklearn.log_model(svc, "my_model_svc_1")
    mlflow.log_artifacts("data/")

    artifacts_uri=mlflow.get_artifact_uri()
    print("The artifact path is",artifacts_uri)

    mlflow.end_run()