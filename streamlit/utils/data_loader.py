"""
=========================================================================
Project:
- Customer Relationship Management

Module:
- utils

File:
- data_loader.py

Purpose:
- Streamlit 화면에서 사용하는 데이터 로드 기능을 제공합니다.

Author: @nobrain711
Created: 2026-03-13

Updated:
- 2026-03-13: initial version (@nobrain711)
- 2026-03-15: MLflow leaderboard 연동 (@nobrain711)
=========================================================================
"""

from pandas import DataFrame
from typing import Dict, List

from common.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from common.mlflow.load_latest_scores_by_model import load_latest_scores_by_model


MODEL_NAME_MAP = {
    "hist_gradient_boosting": "HistGradientBoosting",
    "xgboost_random_grid_search": "XGBoost",
    "easy_ensemble_baseline": "EasyEnsemble",
    "logistic_regression_baseline": "Logistic Regression",
    "lightgbm_baseline": "LightGBM",
}


def get_leaderboard_data() -> DataFrame:
    """
    MLflow에서 최신 모델 리더보드 데이터를 조회하여 반환합니다.
    """
    latest_scores = load_latest_scores_by_model(
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
    )

    rows = []
    for run_name, run_info in latest_scores.items():
        metrics = run_info.get("metrics", {})

        rows.append(
            {
                "모델명(Model)": MODEL_NAME_MAP.get(run_name, run_name),
                "정확도(Acc)": metrics.get("accuracy"),
                "정밀도(Precision)": metrics.get("precision"),
                "재현율(Recall)": metrics.get("recall"),
                "ROC-AUC": metrics.get("roc_auc"),
                "PR-AUC": metrics.get("pr_auc"),
                "F1-score": metrics.get("f1_score"),
            }
        )

    leaderboard_data = DataFrame(rows)

    if leaderboard_data.empty:
        return leaderboard_data

    leaderboard_data = leaderboard_data.sort_values(
        by="ROC-AUC",
        ascending=False,
    ).reset_index(drop=True)

    return leaderboard_data


def get_impact_data() -> DataFrame:
    """
    SHAP 예시용 feature 영향력 데이터를 반환합니다.
    """
    impact_df = DataFrame(
        {
            "feature": [
                "총 결제 금액",
                "소득 구간",
                "총 결제 횟수",
                "한도 소진율",
                "리볼빙 잔액",
            ],
            "영향력": [0.55, 0.45, 0.38, 0.25, 0.18],
        }
    ).sort_values(by="영향력")

    return impact_df


def get_model_guides() -> Dict[str, List[str]]:
    """
    모델별 전략 가이드 데이터를 반환합니다.
    """
    return {
        "HistGradientBoosting": [
            "대용량 데이터 미세 패턴 포착",
            "과적합 위험(Noise)",
            "Random Forest 교차 타겟팅",
        ],
        "XGBoost": [
            "정확도/재현율 밸런스 에이스",
            "복잡한 비선형 해석 난해",
            "Logistic Regression 보조 활용",
        ],
        "EasyEnsemble": [
            "이탈자 전수 탐지(Recall 0.98)",
            "마케팅 비용 낭비 우려",
            "LightGBM 기반 예산 효율화",
        ],
        "Random Forest": [
            "안정적인 일반화 성능",
            "정교한 신호 탐지 한계",
            "HistGBM 모델 고도화 추천",
        ],
        "LightGBM": [
            "실시간 오퍼 최적화 속도",
            "소량 데이터 취약성",
            "XGBoost와 상호 보완",
        ],
        "Logistic Regression": [
            "명확한 원인 설명력",
            "비선형 관계 포착 한계",
            "앙상블 모델 병행 사용",
        ],
    }