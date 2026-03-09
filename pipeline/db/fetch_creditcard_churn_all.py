"""
=========================================================================
Project:
- <project-name>

Module:
- 

File: fetch_creditcard_churn_all.py

Purpose:
- TODO: Describe the purpose of this file

Author: qazx9
Created: 2026-03-08

Updated:
- 2026-03-08: initial version (qazx9)
=========================================================================
"""
import pandas as pd

from .get_select_connection import get_select_connection

def fetch_creditcard_churn_all() -> pd.DataFrame:
    """

    :param query:
    :return:
    """

    query = """
    SELECT *
    FROM creditcard_churn
    """

    conn = get_select_connection()

    try:
        df = pd.read_sql(query, conn)

        return df

    finally:
        conn.close()