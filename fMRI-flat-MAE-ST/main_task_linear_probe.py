import argparse
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

from util.misc import setup_for_distributed

TASK_ID_MAP = {
    "EMOTION": 0,
    "GAMBLING": 1,
    "LANGUAGE": 2,
    "MOTOR": 3,
    "MOVIE1": 4,
    "MOVIE2": 4,
    "MOVIE3": 4,
    "MOVIE4": 4,
    "RELATIONAL": 5,
    "REST1": 6,
    "REST2": 6,
    "REST3": 6,
    "REST4": 6,
    "RETBAR1": 7,
    "RETBAR2": 7,
    "RETCCW": 8,
    "RETCON": 9,
    "RETCW": 10,
    "RETEXP": 11,
    "SOCIAL": 12,
    "WM": 13,
}


def get_args_parser():
    parser = argparse.ArgumentParser("MAE fMRI task linear probe", add_help=False)
    parser.add_argument(
        "--feat_prefix",
        type=str,
        required=True,
        help="feature parquet file prefix",
    )
    parser.add_argument(
        "--output_path",
        default="",
        help="path where to save json result",
    )
    return parser


def main(args):
    setup_for_distributed(True)
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    print("Loading features")
    train_features = pd.read_parquet(f"{args.feat_prefix}_train.parquet")
    test_features = pd.read_parquet(f"{args.feat_prefix}_test.parquet")
    print(f"train: {train_features.shape}, test: {test_features.shape}")

    X_train = np.stack(train_features["feature"])
    X_test = np.stack(test_features["feature"])
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    y_train = train_features["task"].apply(lambda v: TASK_ID_MAP[v]).values
    y_test = test_features["task"].apply(lambda v: TASK_ID_MAP[v]).values
    print(
        f"\ny_train: {y_train.shape} {y_train[:20]}\n"
        f"y_test: {y_test.shape} {y_test[:20]}"
    )
    del train_features, test_features

    train_ind, val_ind = train_test_split(
        np.arange(len(X_train)), train_size=0.9, random_state=42
    )
    print(
        f"\ntrain_ind: {len(train_ind)} {train_ind[:10]}\n"
        f"val_ind: {len(val_ind)} {val_ind[:10]}"
    )
    X_train, X_val = X_train[train_ind], X_train[val_ind]
    y_train, y_val = y_train[train_ind], y_train[val_ind]

    print("Fitting PCA projection")
    pca = PCA(n_components=384, whiten=True, svd_solver="randomized")
    pca.fit(X_train)

    X_train = pca.transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)

    print("Fitting logistic regression")
    clf = LogisticRegressionCV()
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    val_acc = clf.score(X_val, y_val)
    test_acc = clf.score(X_test, y_test)

    result = {
        "feat_prefix": args.feat_prefix,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
    }

    if args.output_path:
        Path(args.output_path).parent.mkdir(exist_ok=True, parents=True)
        with open(args.output_path, "a") as f:
            print(json.dumps(result), file=f)

    print(f"Done:\n{json.dumps(result)}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
