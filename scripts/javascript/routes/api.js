// BRING IN DEPENDENCIES
const express = require('express');
const path = require('path');
var bodyParser = require('body-parser')



// CREATE THE ROUTER OBJECT
const router  = express.Router();



// ROUTES

// When someone hits the python endpoint...
router.post('/possibleAuthors', (req, res) => {

    // Var used to spawn the python script
    const spawn = require('child_process').spawn;

    // Run the python script
    const pythonP = spawn('python', [path.resolve('scripts/python/test.py'), req.body.exerpt]);
    
    // If everything looked good, run the success function
    pythonP.stdout.on('data', function(data) {
        console.log(data.toString());
        res.send(data.toString());
    });
});

// Export the router
module.exports = router;