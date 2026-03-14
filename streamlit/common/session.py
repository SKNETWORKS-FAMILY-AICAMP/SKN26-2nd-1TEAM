"""
=========================================================================
Project:
- Customer Relationship Management

Module:
- common

File:
- session.py

Purpose:
- Streamlit session_state 초기화를 담당합니다.

Author: @nobrain711
Created: 2026-03-13

Updated:
- 2026-03-13: initial version (@nobrain711)
=========================================================================
"""

# streamlit session_stste function
from streamlit import session_state

def init_session_state() -> None:
    """
    앱 실행에 필요한 session_state 기본값을 초기화합니다.
    """
    default_state = {
        "page": "Dashboard",
        "res_prob": 0.0,
        "last_model": "HistGradientBoosting",
    }

    for key, value in default_state.items():
        if key not in session_state:
            session_state[key] = value