// BRING IN DEPENDENCIES
const express = require('express');
const path = require('path');



// CREATE THE ROUTER OBJECT
const router  = express.Router();



// ROUTES

// When someone hits the python endpoint...
router.get('/', (req, res) => {
    res.sendFile(path.resolve('public/index.html'));
});

// Export the router
module.exports = router;