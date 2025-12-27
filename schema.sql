CREATE TABLE IF NOT EXISTS users (
    userId INTEGER PRIMARY KEY,
    churnPred TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    sessionId TEXT PRIMARY KEY,
    userId INTEGER,
    sessionDate INTEGER,
    sessionLength INTEGER,
    sessionEvents INTEGER,
    FOREIGN KEY (userId) REFERENCES users(userId)
);