const { MongoClient } = require('mongodb');

const uri = 'mongodb+srv://odukoya888:1111%40MastDream@cluster0.pxcg5dt.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0';
const client = new MongoClient(uri);

let db;

async function connectDB() {
  if (db) return db;
  await client.connect();
  db = client.db();
  console.log('Connected to MongoDB');
  return db;
}

module.exports = connectDB;
