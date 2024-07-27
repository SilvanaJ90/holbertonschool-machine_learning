-- Write a SQL script that creates a trigger that resets the attribute valid_email only when the email has been changed.

DELIMITER //

CREATE TRIGGER valid_email_update
AFTER UPDATE ON users
FOR EACH ROW
BEGIN
    UPDATE users 
    SET valid_email = valid_email + NEW.email
END //
DELIMITER ;