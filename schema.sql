DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS sessions;

CREATE TABLE users (
    userId INTEGER PRIMARY KEY,
    churnLabel INTEGER
);

CREATE TABLE sessions (
    sessionId TEXT PRIMARY KEY,
    userId INTEGER,
    sessionDate INTEGER,
    sessionLength INTEGER,
    sessionEvents INTEGER,
    FOREIGN KEY (userId) REFERENCES users(userId)
);