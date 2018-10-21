// Create the angular app
var app = angular.module('author', []);

// Add a controller
app.controller('authorCtrl', function($scope, $http) {

    // Get the album objects from the S3 bucket (done server side)
    $scope.getPossibleAuthors = function() {
        
        var data = $.param({
            exerpt: $scope.textModel
        });
    
        var config = {
            headers : {
                'Content-Type': 'application/json;'
            }
        }

        $http.post('http://localhost:3000/api/possibleAuthors', data, config)
            .then(function(response) {
                $scope.authorResponse = response.data;
            });
    }
});