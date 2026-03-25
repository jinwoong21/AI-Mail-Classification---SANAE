-- -----------------------------------------------------
-- 1. 메일 로그 테이블 (최신 코드 동기화 완료)
-- -----------------------------------------------------
DROP TABLE IF EXISTS mail_logs CASCADE;

CREATE TABLE mail_logs (
    id SERIAL PRIMARY KEY,
    
    -- [사용자 및 발신자 정보]
    user_email VARCHAR(100) NOT NULL,   -- 수신자(계정 주인)
    sender_name VARCHAR(100),           -- 보낸 사람 이름
    sender_email VARCHAR(100),          -- 보낸 사람 이메일
    
    -- [메일 본문]
    subject VARCHAR(255),
    content TEXT,
    received_at TIMESTAMP,

    -- [상태 플래그] (Boolean)
    is_read BOOLEAN DEFAULT FALSE,
    is_starred BOOLEAN DEFAULT FALSE,
    is_spam BOOLEAN DEFAULT FALSE,      -- 스팸 여부
    is_complaint BOOLEAN DEFAULT FALSE, -- 민원 여부
    is_archived BOOLEAN DEFAULT FALSE,  -- [필수] 보관 여부 (이거 없으면 보관함 기능 에러남)

    -- [AI 분석 결과]
    priority_level INTEGER DEFAULT 0,   -- 긴급도 (0, 1, 2)
    ai_dept VARCHAR(50),                -- 부서 분류
    ai_assignee_level INTEGER DEFAULT 0,-- 담당자 레벨
    ai_summary TEXT,                    -- 3줄 요약
    ai_confidence INTEGER DEFAULT 0,    -- 신뢰도 (%)
    ai_category VARCHAR(50),            -- 카테고리 (사내업무/거래처 등)
    sentiment_score INTEGER DEFAULT 50, -- 감정 점수 (0~100)

    -- [시스템 관리]
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------------
-- 2. 회원가입 유저 테이블 (기존 유지)
-- -----------------------------------------------------
DROP TABLE IF EXISTS signup_user;

CREATE TABLE signup_user (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    name VARCHAR(100) NOT NULL,
    deptno INT,
    position VARCHAR(50),
    job_level VARCHAR(50),
    role VARCHAR(20) DEFAULT 'USER',
    is_active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);