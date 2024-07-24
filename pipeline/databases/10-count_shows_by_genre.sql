-- Write a script that lists all genres from hbtn_0d_tvshows and displays the number of shows linked to each.

SELECT tg.name AS genre, COUNT(tsg.show_id) AS number_of_shows
FROM tv_show_genres AS tsg
INNER JOIN tv_genres AS tg
ON tsg.genre_id = tg.id
GROUP BY tg.id
ORDER BY  COUNT(tsg.show_id) DESC;
