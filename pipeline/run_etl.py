"""
=========================================================================
Project:
- project-name

Module:
- pipeline

File: run_etl.py

Purpose:
- container for running the entire ETL process

Author: 조동휘
Created: 2026-03-09

Updated:
- 2026-03-09: initial version (조동휘)
=========================================================================
"""
import pandas as pd

from pipeline.db import insert_dataframe
from pipeline.etl import load_csv
from pipeline.etl import standardize_customer_columns

def run_etl() -> int:
    """
    test for etl line

    :return:
    """

    df = load_csv()
    df = standardize_customer_columns(df)

    insert_count = insert_dataframe("creditcard_churn", df)
    
    print(f"Inserted {insert_count} rows into the database.")
    
    return insert_count

if __name__ == '__main__':
    run_etl()