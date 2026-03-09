"""
=========================================================================
Project:
- <project-name>

Module:
- pipeline/db

File: get_select_connection.py

Purpose:
- TODO: Database connection

Author: 조동휘
Created: 2026-03-08

Updated:
- 2026-03-08: initial version (조동휘)
=========================================================================
"""
import pymysql
from .config import DB_SELECT_CONFIG

def get_select_connection():
    """connection for read operations"""
    return pymysql.connect(**DB_SELECT_CONFIG)