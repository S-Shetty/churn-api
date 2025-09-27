# churnexplainer.py

import os
import dill
import pandas as pd
import numpy as np
from sklearn.pipeline import TransformerMixin
from sklearn.preprocessing import LabelEncoder

DATA_DIR = os.path.abspath(".")

class ExplainedModel:
    def __init__(self, labels=None, data=None, categoricalencoder=None, pipeline=None, explainer=None):
        self.data = data
        self.labels = labels
        self.categoricalencoder = categoricalencoder
        self.pipeline = pipeline
        self.explainer = explainer

    @staticmethod
    def load(model_name) -> "ExplainedModel":
        model_dir = os.path.join(DATA_DIR, "models", model_name)
        model_path = os.path.join(model_dir, model_name + ".pkl")
        result = ExplainedModel()
        try:
            with open(model_path, "rb") as f:
                result.__dict__.update(dill.load(f))
            return result
        except OSError as err: 
            print(f"Model path does not exist, returned error: {err}")

    def save(self, model_name):
        model_dir = os.path.join(DATA_DIR, "models", model_name)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_name + ".pkl")
        dilldict = {
            "data": self.data,
            "labels": self.labels,
            "categoricalencoder": self.categoricalencoder,
            "pipeline": self.pipeline,
            "explainer": self.explainer,
        }
        with open(model_path, "wb") as f:
            dill.dump(dilldict, f)

    def predict_df(self, df):
        X = self.categoricalencoder.transform(df)
        return self.pipeline.predict_proba(X)[:, 1]

    def explain_df(self, df):
        X = self.categoricalencoder.transform(df)
        probability = self.pipeline.predict_proba(X)[0, 1]
        e = self.explainer.explain_instance(X[0], self.pipeline.predict_proba).as_map()[1]
        explanations = {self.explainer.feature_names[c]: weight for c, weight in e}
        return probability, explanations

    def explain_dct(self, dct):
        return self.explain_df(pd.DataFrame([dct]))

    def cast_dct(self, dct):
        result = {}
        for k, v in dct.items():
            dtype = self.dtypes[k]
            try:
                if pd.api.types.is_categorical_dtype(dtype):
                    result[k] = v
                else:
                    result[k] = dtype.type(v)
            except Exception:
                result[k] = v
        return result

    @property
    def dtypes(self):
        if not hasattr(self, "_dtypes"):
            d = self.data[self.non_categorical_features].dtypes.to_dict()
            d.update({c: self.data[c].cat.categories.dtype for c in self.categorical_features})
            self._dtypes = d
        return self._dtypes

    @property
    def non_categorical_features(self):
        return list(
            self.data.select_dtypes(exclude=["category"]).columns.drop(self.labels.name + " probability")
        )

    @property
    def categorical_features(self):
        return list(self.data.select_dtypes(include=["category"]).columns)

    @property
    def stats(self):
        def describe(s):
            return {
                "median": s.median(),
                "mean": s.mean(),
                "min": s.min(),
                "max": s.max(),
                "std": s.std(),
            }
        if not hasattr(self, "_stats"):
            self._stats = {
                c: describe(self.data[c]) for c in self.non_categorical_features
            }
        return self._stats

    @property
    def label_name(self):
        return self.labels.name + " probability"

    @property
    def categories(self):
        return {
            feature: list(self.categoricalencoder.classes_[feature])
            for feature in self.categorical_features
        }

    @property
    def default_data(self):
        if not hasattr(self, "_default_data"):
            d = {}
            d.update({
                feature: self.categoricalencoder.classes_[feature][0]
                for feature in self.categorical_features
            })
            d.update({
                feature: self.data[feature].median()
                for feature in self.non_categorical_features
            })
            self._default_data = d
        return self._default_data

class CategoricalEncoder(TransformerMixin):
    def fit(self, X, y=None, *args, **kwargs):
        self.columns_ = X.columns
        self.cat_columns_ix_ = {
            c: i for i, c in enumerate(X.columns) if pd.api.types.is_categorical_dtype(X[c])
        }
        self.cat_columns_ = pd.Index(self.cat_columns_ix_.keys())
        self.non_cat_columns_ = X.columns.drop(self.cat_columns_)
        self.les_ = {c: LabelEncoder().fit(X[c]) for c in self.cat_columns_}
        self.classes_ = {c: list(self.les_[c].classes_) for c in self.cat_columns_}
        return self

    def transform(self, X, y=None, *args, **kwargs):
        data = X[self.columns_].values
        for c, i in self.cat_columns_ix_.items():
            data[:, i] = self.les_[c].transform(data[:, i])
        return data.astype(float)

    def __repr__(self):
        return f"{self.__class__.__name__}()"
