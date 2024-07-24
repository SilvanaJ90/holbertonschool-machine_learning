-- Write a script that lists all genres in the database hbtn_0d_tvshows_rate by their rating.

SELECT tg.name, SUM(tsr.rate) AS rating
FROM tv_genres AS tg
JOIN tv_show_genres AS tsg
ON tg.id = tsg.genre_id
JOIN tv_shows AS ts
ON ts.id = tsg.show_id
JOIN tv_show_ratings AS tsr
ON ts.id = tsr.show_id
GROUP BY tg.name
ORDER BY rating DESC;
