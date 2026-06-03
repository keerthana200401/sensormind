CREATE DATABASE IF NOT EXISTS sensormind;
USE sensormind;

CREATE TABLE IF NOT EXISTS predictions (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    engine_id       VARCHAR(50)     NOT NULL,
    rul_prediction  FLOAT           NOT NULL,
    risk_level      VARCHAR(20)     NOT NULL,
    is_anomaly      BOOLEAN         NOT NULL,
    anomaly_score   FLOAT,
    sensor_13       FLOAT,
    sensor_15       FLOAT,
    sensor_11       FLOAT,
    model_used      VARCHAR(50),
    created_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS alerts (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    engine_id       VARCHAR(50)     NOT NULL,
    prediction_id   INT,
    alert_text      TEXT            NOT NULL,
    risk_level      VARCHAR(20)     NOT NULL,
    similar_failures VARCHAR(200),
    input_tokens    INT,
    output_tokens   INT,
    created_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
);

CREATE TABLE IF NOT EXISTS sensor_stream (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    engine_id       VARCHAR(50)     NOT NULL,
    cycle           INT             NOT NULL,
    sensor_13       FLOAT,
    sensor_15       FLOAT,
    sensor_11       FLOAT,
    raw_data        JSON,
    created_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP
);