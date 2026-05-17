-- PostgreSQL init script: create multiple databases
SELECT 'CREATE DATABASE mlflow_db' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow_db')\gexec
SELECT 'CREATE DATABASE airflow_db' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow_db')\gexec
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlops;
GRANT ALL PRIVILEGES ON DATABASE airflow_db TO mlops;
