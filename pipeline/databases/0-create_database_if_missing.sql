-- Create database + user if doesn't exist
CREATE DATABASE IF NOT EXISTS db_0;
CREATE USER IF NOT EXISTS 'db_0_dev'@'localhost' IDENTIFIED BY 'db_0_pwd';
GRANT ALL ON `db_0`.* TO 'db_0_dev'@'localhost';
GRANT SELECT ON `performance_schema`.* TO 'db_0_dev'@'localhost';
FLUSH PRIVILEGES;
