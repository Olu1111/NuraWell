document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.getElementById('login-form');
    const signupForm = document.getElementById('signup-form');
    const chatForm = document.getElementById('chat-form');
    const showSignup = document.getElementById('show-signup');
    const showLogin = document.getElementById('show-login');
    const authContainer = document.getElementById('auth-container');
    const signupContainer = document.getElementById('signup-container');
    const chatContainer = document.getElementById('chat-container');
    const chatHistory = document.getElementById('chat-history');
    const chatInput = document.getElementById('chat-input');
    const sidebar = document.getElementById('sidebar');
    const toggleSidebarBtn = document.getElementById('toggle-sidebar'); // NEW toggle button
    const closeSidebarBtn = document.getElementById('close-sidebar');
    const newChatBtn = document.getElementById('new-chat');

    let token = null;

    // ===== Auth Navigation =====
    showSignup.addEventListener('click', (e) => {
        e.preventDefault();
        authContainer.style.display = 'none';
        signupContainer.style.display = 'block';
    });

    showLogin.addEventListener('click', (e) => {
        e.preventDefault();
        signupContainer.style.display = 'none';
        authContainer.style.display = 'block';
    });

    // ===== Sidebar Controls =====
    // Toggle sidebar open/close from main screen
    toggleSidebarBtn.addEventListener('click', () => {
        sidebar.classList.toggle('hidden');
    });

    // Open sidebar from "New Chat" button
    newChatBtn.addEventListener('click', () => {
        sidebar.classList.remove('hidden');
    });

    // Close sidebar from X button
    closeSidebarBtn.addEventListener('click', () => {
        sidebar.classList.add('hidden');
    });

    // ===== Signup =====
    signupForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = document.getElementById('signup-email').value;
        const password = document.getElementById('signup-password').value;

        const res = await fetch('/api/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password }),
        });

        if (res.ok) {
            alert('Signup successful! Please login.');
            showLogin.click();
        } else {
            const data = await res.json();
            alert(data.msg);
        }
    });

    // ===== Login =====
    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = document.getElementById('login-email').value;
        const password = document.getElementById('login-password').value;

        const res = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password }),
        });

        if (res.ok) {
            const data = await res.json();
            token = data.token;

            // Hide auth forms
            authContainer.style.display = 'none';
            signupContainer.style.display = 'none';

            // Show chat UI but keep sidebar hidden until user toggles
            chatContainer.style.display = 'flex';
            sidebar.classList.add('hidden'); // sidebar stays hidden until opened
        } else {
            const data = await res.json();
            alert(data.msg);
        }
    });

    // ===== Chat Submission =====
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const message = chatInput.value;
        if (!message) return;

        appendMessage(message, 'user-message');
        chatInput.value = '';

        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({ message }),
        });

        const data = await res.json();
        appendMessage(data.response, 'bot-message');
    });

    // ===== Append Message to Chat History =====
    function appendMessage(message, className) {
        const messageElement = document.createElement('div');
        messageElement.classList.add(className);
        messageElement.innerText = message;
        chatHistory.appendChild(messageElement);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
});
