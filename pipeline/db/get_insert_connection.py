"""
=========================================================================
Project:
- <project-name>

Module:
- 

File: get_insert_connection.py

Purpose:
- TODO: Describe the purpose of this file

Author: qazx9
Created: 2026-03-08

Updated:
- 2026-03-08: initial version (qazx9)
=========================================================================
"""
import pymysql

from .config import DB_INSERT_CONFIG

def get_insert_connection():
    """connection for write operations"""
    return pymysql.connect(**DB_INSERT_CONFIG)