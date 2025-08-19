const express = require('express');
const router = express.Router();
const axios = require('axios');

// Configuration
const PYTHON_SERVICE_URL = 'http://localhost:8000/chat';
const TIMEOUT = 30000; // 30 seconds

router.post('/', async (req, res) => {
  try {
    const { message } = req.body;
    
    // Add timeout to frontend request
    const response = await axios.post('http://localhost:8000/chat', {
      message: message,
      timeout: 45  // Give more time than service timeout
    }, {
      timeout: 50000  // Higher than both timeouts
    });
    
    res.json(response.data);
    
  } catch (error) {
    if (error.code === 'ECONNABORTED') {
      res.status(504).json({ error: 'Request timeout', suggestion: 'Try a shorter message' });
    } else if (error.response?.data?.detail?.includes('timeout')) {
      res.status(504).json({ 
        error: 'Processing timeout',
        suggestion: 'The model needs more time, try reducing max_new_tokens'
      });
    } else {
      console.error('Service error:', error);
      res.status(500).json({ 
        error: 'Service error',
        details: error.response?.data || error.message
      });
    }
  }
});

module.exports = router;