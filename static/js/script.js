function updateImage(input) {
    if (input.files && input.files[0]) {
        var img = document.getElementById('skin_image');
        img.src = URL.createObjectURL(input.files[0]);
        img.onload = function() {
            sendImageToServer(input.files[0]);
            URL.revokeObjectURL(img.src); // To free up resources
        }
    }
}

function sendImageToServer(file) {
    var formData = new FormData();
    formData.append('image', file);

    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/predict', true);
    xhr.onload = function() {
        if (xhr.status === 200) {
            var data = JSON.parse(xhr.responseText);

            var entries = Object.entries(data);

            entries.sort(function(a, b) {
                return b[1] - a[1];
            });

            data = Object.fromEntries(entries);

            console.log(data);
            var result_string = "Results: <br>"
            var i = 0;
            for (var key in data) {
                if (i >= 4) {
                    break;
                }
                result_string += key + ": " + data[key] + "%" + "<br>";
                i++;
            }

            document.getElementById('result').innerHTML = result_string;
            // document.querySelector('.result-box').style.visibility = 'visible';
            console.log('Image uploaded successfully');
        } else {
            console.error('Error uploading image');
        }
    };
    xhr.send(formData);
}