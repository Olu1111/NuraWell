-- Create the users table
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE,
    phone_number VARCHAR(20),
    password_hash VARCHAR(255),
    profile_picture_url TEXT,
    is_guest BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create the user_settings table
CREATE TABLE user_settings (
    user_id UUID PRIMARY KEY REFERENCES users(id),
    notifications_enabled BOOLEAN DEFAULT TRUE,
    chatbot_preferences JSONB,
    language VARCHAR(10) DEFAULT 'en-US',
    color_scheme VARCHAR(20) DEFAULT 'light'
);

-- Create the conversations table
CREATE TABLE conversations (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    title VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create the messages table
CREATE TABLE messages (
    id UUID PRIMARY KEY,
    conversation_id UUID REFERENCES conversations(id),
    sender VARCHAR(10) NOT NULL, -- 'user' or 'bot'
    content_type VARCHAR(20) NOT NULL, -- 'text', 'image', 'audio'
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
