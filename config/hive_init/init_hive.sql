-- 데이터베이스 생성
CREATE DATABASE IF NOT EXISTS hive_mail_db;

-- DB 선택
USE hive_mail_db;

-- 테이블 초기화
DROP TABLE IF EXISTS mail_logs;

-- Hive 테이블 생성 (Postgres 스키마와 1:1 매칭)
CREATE TABLE mail_logs (
    id BIGINT,
    
    -- [사용자 및 발신자 정보]
    user_email STRING,
    sender_name STRING,
    sender_email STRING,
    
    -- [메일 본문]
    subject STRING,
    content STRING,
    received_at TIMESTAMP,

    -- [상태 플래그]
    is_read BOOLEAN,
    is_starred BOOLEAN,
    is_spam BOOLEAN,
    is_complaint BOOLEAN,
    is_archived BOOLEAN,  -- [추가됨] 보관 여부

    -- [AI 분석 결과]
    priority_level INT,
    ai_dept STRING,
    ai_assignee_level INT,
    ai_summary STRING,
    ai_confidence INT,
    ai_category STRING,
    sentiment_score INT,

    -- [시스템 관리]
    created_at TIMESTAMP
)
STORED AS PARQUET;