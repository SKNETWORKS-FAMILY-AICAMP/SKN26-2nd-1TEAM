"""
=========================================================================
Project:
- <project-name>

Module:
- 

File: dataframe_to_tuples.py

Purpose:
- TODO: Describe the purpose of this file

Author: qazx9
Created: 2026-03-08

Updated:
- 2026-03-08: initial version (qazx9)
=========================================================================
"""
from typing import List, Tuple

import pandas as pd

def dataframe_to_tuples(df: pd.DataFrame) -> List[Tuple]:
    """
    convert pandas dataframe to list of tuples

    :param df:
    :return:
    """

    return [tuple(row) for row in df.to_numpy()]