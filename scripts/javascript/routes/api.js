// BRING IN DEPENDENCIES
const express = require('express');
const path = require('path');
var http = require('http');
var requestify = require('requestify');
const bodyParser = require('body-parser');

// CREATE THE ROUTER OBJECT
const router  = express.Router();



// ROUTES

// When someone hits the python endpoint...
router.post('/possibleAuthors', function(req, res) {
	console.log(req.body);
	requestify.post('http://localhost:5000/confidence', {
		method: 'POST',
		body: {
    			exerpt: req.body.exerpt
		},
		headers: {
			'Content-Type': 'application/json'
		},
		datatype: 'json'
	}).then(function(response) {
	    res.send(response.getBody());
	});
});

// Export the router
module.exports = router;
