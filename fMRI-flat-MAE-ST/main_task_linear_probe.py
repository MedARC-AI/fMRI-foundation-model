import argparse
import json
import os
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from util.misc import setup_for_distributed


def get_args_parser():
    parser = argparse.ArgumentParser("MAE fMRI linear probe", add_help=False)
    parser.add_argument(
        "--feat_prefix",
        type=str,
        required=True,
        help="feature parquet file prefix",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="task",
        choices=["task", "trial_type"],
        help="prediction target",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="path where to save, empty for no saving",
    )
    return parser


def main(args):
    setup_for_distributed(True)
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    else:
        output_dir = None

    print("Loading features")
    train_features = pd.read_parquet(f"{args.feat_prefix}_train.parquet")
    test_features = pd.read_parquet(f"{args.feat_prefix}_test.parquet")
    print(f"train: {train_features.shape}, test: {test_features.shape}")

    X_train = np.stack(train_features["feature"])
    X_test = np.stack(test_features["feature"])
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    if args.target == "task":
        labels_train = train_features["task"].str.rstrip("1234").values
        labels_test = test_features["task"].str.rstrip("1234").values
    elif args.target == "trial_type":
        labels_train = train_features["trial_type"].values
        labels_test = test_features["trial_type"].values

    label_enc = LabelEncoder()
    y_train = label_enc.fit_transform(labels_train)
    y_test = label_enc.transform(labels_test)

    print(f"classes ({len(label_enc.classes_)}): {label_enc.classes_}")
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
        "target": args.target,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
    }

    if output_dir:
        print(f"Saving results to: {output_dir}")
        with open(output_dir / "result.json", "w") as f:
            print(json.dumps(result), file=f)

        state = {
            "label_enc": label_enc,
            "pca": pca,
            "clf": clf,
            "result": result,
        }
        with open(output_dir / "state.pkl", "wb") as f:
            pickle.dump(state, f)

    print(f"Done:\n{json.dumps(result)}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
