-- Write a SQL script that creates a stored procedure ComputeAverageWeightedScoreForUser that computes and store the average weighted score for a student.

-- Requirements:

    -- Procedure ComputeAverageScoreForUser is taking 1 input:
        -- user_id, a users.id value (you can assume user_id is linked to an existing users)

-- Tips:

    -- Calculate-Weighted-Average https://www.wikihow.com/Calculate-Weighted-Average


DELIMITER $$

CREATE PROCEDURE ComputeAverageWeightedScoreForUser (IN user_id INT)
BEGIN
    DECLARE weighted_avg FLOAT;
    DECLARE total_weight INT;

    -- Calculate the weighted average score
    SELECT SUM(c.score * p.weight) / SUM(p.weight) INTO weighted_avg
    FROM corrections c
    JOIN projects p ON c.project_id = p.id
    WHERE c.user_id = user_id;
    
    -- Update the average score in the users table
    UPDATE users
    SET average_score = weighted_avg
    WHERE id = user_id;
END$$

DELIMITER ;
