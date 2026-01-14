DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS sessions;

CREATE TABLE users (
    userId INTEGER PRIMARY KEY,
    churnLabel INTEGER CHECK (churnLabel IN (0,1))
);

CREATE TABLE sessions (
    sessionId TEXT PRIMARY KEY,
    userId INTEGER NOT NULL, 
    sessionDate INTEGER NOT NULL,
    sessionLength INTEGER NOT NULL,
    sessionEvents INTEGER NOT NULL, 
    FOREIGN KEY (userId) REFERENCES users(userId)
);