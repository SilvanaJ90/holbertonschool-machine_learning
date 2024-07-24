-- create db and user

CREATE DATABASE IF NOT EXISTS names;
CREATE USER IF NOT EXISTS 'names_dev'@'localhost' IDENTIFIED BY 'names_pwd';
GRANT ALL ON `names`.* TO 'names_dev'@'localhost';
GRANT SELECT ON `performance_schema`.* TO 'names_dev'@'localhost';
FLUSH PRIVILEGES;


-- import db

sudo mysql -u names_dev  -p names < names.sql