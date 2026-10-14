DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS sessions;
DROP TABLE IF EXISTS feature_usage;
DROP TABLE IF EXISTS session_events;

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

-- Optional: per-session feature usage (enables depth/breadth/drop-off analytics)
CREATE TABLE feature_usage (
    userId INTEGER NOT NULL,
    sessionId TEXT NOT NULL,
    sessionDate INTEGER NOT NULL,
    featureKey TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (userId) REFERENCES users(userId),
    FOREIGN KEY (sessionId) REFERENCES sessions(sessionId)
);

-- Optional: event-level table (future expansion; not used by v1 feature pipeline)
CREATE TABLE session_events (
    eventId TEXT PRIMARY KEY,
    sessionId TEXT NOT NULL,
    userId INTEGER NOT NULL,
    eventTime TEXT,
    eventType TEXT,
    featureKey TEXT,
    FOREIGN KEY (userId) REFERENCES users(userId),
    FOREIGN KEY (sessionId) REFERENCES sessions(sessionId)
);