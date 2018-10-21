// Create the angular app
var app = angular.module('author', []);

// Add a controller
app.controller('authorCtrl', function($scope, $http) {


    $scope.update = function() {
	console.log('ez');
	console.log($scope.authorResponse);
    }

    // Get the album objects from the S3 bucket (done server side)
    $scope.getPossibleAuthors = function() {
        
	console.log($scope.textModel);

	$http({
  		method  : 'POST',
  		url     : 'http://localhost:3000/api/possibleAuthors',
  		data    : {'exerpt':$scope.textModel},  // pass in data as strings
  		headers : { 'Content-Type': 'application/json' }  // set the headers so angular passing info as form data (not request payload)
 	}).then(function(response) {
		console.log(response.data);
		$scope.authorResponse = response.data;
	}).then($scope.update);
    }

});
