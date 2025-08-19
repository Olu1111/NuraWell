const express = require('express');
const connectDB = require('./db');
require('dotenv').config();
const app = express();
const port = process.env.PORT || 3000;

app.use(express.json());

app.use(express.static('public'));

app.get('/', (req, res) => {
  res.sendFile(__dirname + '/public/index.html');
});

app.get('/test-db', async (req, res) => {
  try {
    const db = await connectDB();
    const result = await db.admin().ping();
    res.json(result);
  } catch (err) {
    console.error(err);
    res.status(500).send('Error connecting to the database');
  }
});

// Define Routes
app.use('/api/auth', require('./routes/auth'));
app.use('/api/chat', require('./routes/chat'));

connectDB().then(() => {
  app.listen(port, () => {
    console.log(`Server is listening on port ${port}`);
  });
});
