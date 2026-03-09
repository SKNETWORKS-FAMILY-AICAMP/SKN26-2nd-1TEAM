"""
=========================================================================
Project:
- <project-name>

Module:
- 

File: build_insert_query.py

Purpose:
- TODO: Describe the purpose of this file

Author: qazx9
Created: 2026-03-08

Updated:
- 2026-03-08: initial version (qazx9)
=========================================================================
"""
from typing import List


def build_insert_query(table_name: str, columns: List[str]) -> str:
    """
    build INSERT query

    :param table_name:
    :param columns:
    :return:
    """

    cols = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))

    return f"""
        INSERT INTO {table_name} 
        ({cols}) 
        VALUES ({placeholders})
    """.strip()