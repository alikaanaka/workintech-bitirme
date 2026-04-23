"""Categorical encoding utilities."""

from dataclasses import dataclass
import re

import pandas as pd


@dataclass
class EncodedFoldData:
    """Container for encoded fold data."""

    train_x: pd.DataFrame
    valid_x: pd.DataFrame


class FoldEncoder:
    """Fit fold-local categorical mappings and transform consistently."""

    def __init__(self) -> None:
        self._categorical_columns: list[str] = []
        self._numerical_columns: list[str] = []
        self._category_maps: dict[str, dict[str, int]] = {}

    @staticmethod
    def _sanitize_columns(columns: list[str]) -> list[str]:
        """Sanitize column names for LightGBM JSON-safe feature names."""
        sanitized = []
        seen: dict[str, int] = {}
        for column in columns:
            base_name = re.sub(r"[^A-Za-z0-9_]", "_", str(column))
            base_name = re.sub(r"_+", "_", base_name).strip("_")
            base_name = base_name or "feature"
            count = seen.get(base_name, 0)
            if count > 0:
                clean_name = f"{base_name}_{count}"
            else:
                clean_name = base_name
            seen[base_name] = count + 1
            sanitized.append(clean_name)
        return sanitized

    def fit(self, train_x: pd.DataFrame) -> "FoldEncoder":
        """Fit label maps on categorical columns of training data."""
        self._categorical_columns = train_x.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        self._numerical_columns = [column for column in train_x.columns if column not in self._categorical_columns]
        if self._categorical_columns:
            categorical_train = train_x[self._categorical_columns].fillna("MISSING").astype(str)
            self._category_maps = {}
            for column in self._categorical_columns:
                unique_values = sorted(categorical_train[column].unique().tolist())
                self._category_maps[column] = {value: idx for idx, value in enumerate(unique_values)}
        return self

    def transform(self, data_x: pd.DataFrame) -> pd.DataFrame:
        """Transform data into numeric matrix for model training."""
        numeric_part = data_x[self._numerical_columns].copy()
        if not self._categorical_columns:
            numeric_part.columns = self._sanitize_columns(numeric_part.columns.tolist())
            return numeric_part

        categorical_input = data_x[self._categorical_columns].fillna("MISSING").astype(str)
        encoded_df = pd.DataFrame(index=data_x.index)
        for column in self._categorical_columns:
            mapping = self._category_maps.get(column, {})
            encoded_df[f"{column}_ENC"] = categorical_input[column].map(mapping).fillna(-1).astype(float)
        combined = pd.concat([numeric_part, encoded_df], axis=1)
        combined.columns = self._sanitize_columns(combined.columns.tolist())
        return combined

    def fit_transform_fold(self, train_x: pd.DataFrame, valid_x: pd.DataFrame) -> EncodedFoldData:
        """Fit on train and transform both train and validation data."""
        self.fit(train_x)
        return EncodedFoldData(train_x=self.transform(train_x), valid_x=self.transform(valid_x))
