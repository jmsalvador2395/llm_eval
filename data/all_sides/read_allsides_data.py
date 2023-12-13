"""
Author: Naman Bansal
Usage:
    Read Train and Val Data - read_gold_all_sides_data_as_train_val(path_to_train_data_excel)
    Read Test Data - read_all_sides_test_dataset(path_to_test_file)
"""
from enum import Enum
from pathlib import Path
from typing import Union, List, Tuple

import pandas as pd

DF = pd.DataFrame
MODEL_INPUT_DATA_COLS = ["S1", "S2", "S_int"]
ALL_SIDES_ANNOTATOR_COLS = [
    "Ahmed_Intersection",
    "Naman_Intersection",
    "Helen_Intersection",
    "AllSides_Intersection",
]


class DataCategory(Enum):
    AllSides = "all_sides"
    Privacy = "privacy"


class PathOutType(Enum):
    STR = "str"
    PATH = "path"


def full_path(
        inp_dir_or_path: str, out_type: PathOutType = PathOutType.STR
) -> Union[Path, str]:
    """Returns full path"""
    out_pt = Path(inp_dir_or_path).expanduser().resolve()
    if out_type.value == "str":
        return str(out_pt)
    else:
        return out_pt


def select_cols(df: DF, cols: List[str] = MODEL_INPUT_DATA_COLS) -> DF:
    return df[cols]


def read_all_sides_crawled_data(excel_path: str) -> DF:
    """Reads all sides data as dataframe"""
    assert Path(excel_path).suffix == ".xlsx", "Input file in not a excel file"
    df_orig_train = pd.read_excel(
        excel_path, engine="openpyxl"
    )  # engine is required because pandas version is old
    df_orig_train = df_orig_train[df_orig_train["Flag"] == 0]
    df_orig_train = df_orig_train.rename(
        columns={
            "left-context": "S1",
            "right-context": "S2",
            "theme-description": "S_int",
        }
    )
    return select_cols(df_orig_train)


def read_gold_all_sides_data_as_train_val(
        input_path: str,
        data_type: DataCategory,
        n_train: int = 2000,
) -> (DF, DF):
    """Divides the AllSides data into Train and Val dataframes"""
    input_path = full_path(input_path)
    assert Path(input_path).is_file(), "Input path must exist"
    # Load data
    df_orig = read_all_sides_crawled_data(input_path, data_type)  # 2721 samples
    # Randomize with seed=0 (keep things similar to synthetic paper)
    df_orig = df_orig.sample(frac=1, random_state=0).reset_index(drop=True)
    df_train: DF = df_orig.iloc[:n_train, :].reset_index(drop=True)
    df_val: DF = df_orig.iloc[n_train:, :].reset_index(drop=True)
    return df_train, df_val


def read_all_sides_test_dataset(
        pt: Path = Path(
            "~/text_summarization/src/all_sides_gold_data/test.xlsx"
        ),
) -> Tuple[DF, DF]:
    """Reads the AllSides test data and returns the input and ground truth dfs"""
    ref_df = pd.read_excel(full_path(pt))
    ref_df = ref_df.rename(
        columns={
            "Left": "S1",
            "Right": "S2",
        }
    )

    ref_df["S_int"] = ""

    return select_cols(ref_df, MODEL_INPUT_DATA_COLS), select_cols(
        ref_df, ALL_SIDES_ANNOTATOR_COLS
    )
