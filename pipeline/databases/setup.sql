-- create db and user

CREATE DATABASE IF NOT EXISTS hbtn_0d_tvshows_rate;
CREATE USER IF NOT EXISTS 'hbtn_0d_tvshows_rate_dev'@'localhost' IDENTIFIED BY 'hbtn_0d_tvshows_rate_pwd';
GRANT ALL ON `hbtn_0d_tvshows_rate`.* TO 'hbtn_0d_tvshows_rate_dev'@'localhost';
GRANT SELECT ON `performance_schema`.* TO 'hbtn_0d_tvshows_rate_dev'@'localhost';
FLUSH PRIVILEGES;


-- import db

sudo mysql -u hbtn_0d_tvshows_rate_dev  -p hbtn_0d_tvshows_rate < hbtn_0d_tvshows_rate.sql