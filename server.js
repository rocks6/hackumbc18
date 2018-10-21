// SETUP

// Bring in our dependencies
const path    = require('path');
const express = require('express');
var bodyParser = require('body-parser');

// Bring in our routes
const api     = require('./scripts/javascript/routes/api');
const static  = require('./scripts/javascript/routes/static');

// Create an app variable
const app     = express();



// USE THE ROUTES

// Make sure that the public directory is accessible
app.use(express.static(path.join(__dirname, '/public')));
app.use(bodyParser.json());

// Initialize routes that have been created
app.use(static);
app.use('/api', api);



// TURN ON THE SERVER
app.listen(3000, () => {
    console.log('App listening on port 3000');
});