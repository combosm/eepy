var socket = io();
var DATA_POLL_INTERVAL_MS = 3000;

socket.on("update_data", function(data) {
    document.getElementById("ear_value").innerText = data.EAR;
    document.getElementById("mar_value").innerText = data.MAR;
    document.getElementById("drowsy_value").innerText =
        "Drowsiness Status: " + (data.is_drowsy ? "Yes 💤" : "No ✅");
    document.getElementById("ai_response").innerText = data.ai_response;
});

function requestData() {
    fetch('/data')
        .then(response => response.json())
        .then(data => {
            document.getElementById("ear_value").innerText = data.EAR;
            document.getElementById("mar_value").innerText = data.MAR;
            document.getElementById("drowsy_value").innerText =
                "Drowsiness Status: " + (data.is_drowsy ? "Yes 💤" : "No ✅");
            document.getElementById("ai_response").innerText = data.ai_response;
        })
        .catch(error => console.error('Error fetching data:', error));
}

setInterval(requestData, DATA_POLL_INTERVAL_MS);

function fetchAIOutput() {
    fetch('/ai_output')
        .then(response => response.json())
        .then(data => {
            console.log("AI Output:", data.message);  // Console log for debugging
            document.getElementById("ai_response").innerText = data.message;
        })
        .catch(error => console.error('Error fetching AI output:', error));
}
