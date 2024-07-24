--- Write a script that creates a table called first_table in the current database in your MySQL server.

USE db_0;
CREATE TABLE IF NOT EXISTS first_table (
    id INT PRIMARY KEY,
    name VARCHAR(256)
);