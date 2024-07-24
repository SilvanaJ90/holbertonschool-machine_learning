-- create db and user

CREATE DATABASE IF NOT EXISTS metal_bands;
CREATE USER IF NOT EXISTS 'metal_bands_dev'@'localhost' IDENTIFIED BY 'metal_bands_pwd';
GRANT ALL ON `metal_bands`.* TO 'metal_bands_dev'@'localhost';
GRANT SELECT ON `performance_schema`.* TO 'metal_bands_dev'@'localhost';
FLUSH PRIVILEGES;


-- import db

sudo mysql -u metal_bands_dev  -p metal_bands < metal_bands.sql