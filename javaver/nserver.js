const express = require('express');
const path = require('path');
const fileUpload = require('express-fileupload');
const fs = require('fs');
const app = express();
const port = 3000;

// Use express-fileupload middleware
app.use(fileUpload());

// Serve static files (for frontend)
app.use(express.static(path.join(__dirname, 'public')));

// Handle file upload
app.post('/upload', (req, res) => {
  if (!req.files || Object.keys(req.files).length === 0) {
    return res.status(400).send('No files were uploaded.');
  }

  let uploadedFile = req.files.uploadedFile;

  // Save the file to a temporary location
  const filePath = path.join(__dirname, 'uploads', uploadedFile.name);
  uploadedFile.mv(filePath, (err) => {
    if (err) {
      return res.status(500).send(err);
    }

    // Parse the file to extract network data (assuming it's a JSON file)
    fs.readFile(filePath, 'utf8', (err, data) => {
      if (err) {
        return res.status(500).send(err);
      }
      
      const networkData = JSON.parse(data);  // Assuming JSON format
      // Send network data to frontend
      res.json(networkData);
    });
  });
});

// Start the server
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});

