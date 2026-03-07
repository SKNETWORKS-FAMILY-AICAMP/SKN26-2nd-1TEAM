/* =========================================================================
Project:
- (직접 입력)

Module:
- init

File: 006_add_constraints

Purpose:
- 각 tables에 constrains 부여

Author: 조동휘
Created: 2026-03-07

Updated:
- 2026-03-07: initial version
=========================================================================
*/

-- -------------------------------------------------
-- MIGRATION INFO
-- id: (예: 2026-02-09_01_init_schema)
-- apply: up section 실행
-- rollback: down section 실행(가능한 범위에서)
-- -------------------------------------------------

/* =======================
   UP
======================= */
START TRANSACTION;

USE creditcard_churn_db;

-- =========================
-- PRIMARY KEY
-- =========================

ALTER TABLE dim_education
    ADD CONSTRAINT pk_dim_education
        PRIMARY KEY (education_id);

ALTER TABLE dim_marital
    ADD CONSTRAINT pk_dim_marital
        PRIMARY KEY (marital_id);

ALTER TABLE dim_income
    ADD CONSTRAINT pk_dim_income
        PRIMARY KEY (income_id);

ALTER TABLE dim_card_type
    ADD CONSTRAINT pk_dim_card_type
        PRIMARY KEY (card_type_id);

ALTER TABLE creditcard_churn
    MODIFY creditcard_churn_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    ADD CONSTRAINT pk_creditcard_churn
        PRIMARY KEY (creditcard_churn_id);


-- =========================
-- UNIQUE
-- =========================

ALTER TABLE dim_education
    ADD CONSTRAINT uk_dim_education_label
        UNIQUE (education_label);

ALTER TABLE dim_marital
    ADD CONSTRAINT uk_dim_marital_label
        UNIQUE (marital_label);

ALTER TABLE dim_income
    ADD CONSTRAINT uk_dim_income_label
        UNIQUE (income_label);

ALTER TABLE dim_card_type
    ADD CONSTRAINT uk_dim_card_type_label
        UNIQUE (card_type_label);


-- =========================
-- FOREIGN KEY
-- =========================

ALTER TABLE creditcard_churn
    ADD CONSTRAINT fk_creditcard_churn_education
        FOREIGN KEY (education_id)
            REFERENCES dim_education (education_id);

ALTER TABLE creditcard_churn
    ADD CONSTRAINT fk_creditcard_churn_marital
        FOREIGN KEY (marital_id)
            REFERENCES dim_marital (marital_id);

ALTER TABLE creditcard_churn
    ADD CONSTRAINT fk_creditcard_churn_income
        FOREIGN KEY (income_id)
            REFERENCES dim_income (income_id);

ALTER TABLE creditcard_churn
    ADD CONSTRAINT fk_creditcard_churn_card_type
        FOREIGN KEY (card_type_id)
            REFERENCES dim_card_type (card_type_id);


-- =========================
-- CHECK
-- =========================

ALTER TABLE creditcard_churn
    ADD CONSTRAINT chk_creditcard_churn_churn_status
        CHECK (churn_status IN (0, 1));

ALTER TABLE creditcard_churn
    ADD CONSTRAINT chk_creditcard_churn_gender
        CHECK (gender IN (0, 1));

-- =========================
-- CHECK
-- =========================
CREATE INDEX idx_creditcard_churn_education
    ON creditcard_churn (education_id);

CREATE INDEX idx_creditcard_churn_marital
    ON creditcard_churn (marital_id);

CREATE INDEX idx_creditcard_churn_income
    ON creditcard_churn (income_id);

CREATE INDEX idx_creditcard_churn_card_type
    ON creditcard_churn (card_type_id);

COMMIT;


/* =======================
   DOWN
======================= */
-- 주의: 운영에서는 DOWN이 위험할 수 있음. 필요할 때만 작성.
-- START TRANSACTION;
-- DROP TABLE IF EXISTS ...
-- COMMIT;
