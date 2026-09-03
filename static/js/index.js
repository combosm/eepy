var socket = io();
var DATA_POLL_INTERVAL_MS = 3000;

function formatCalibrationReason(reason) {
    return reason
        .split("_")
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(" ");
}

function updateCalibrationDiagnostic(data) {
    var qualityElement = document.getElementById("calibration_quality");
    var reasonsElement = document.getElementById("calibration_reasons");
    var quality = data.calibration_frame_quality;
    var supportedQualities = ["valid", "degraded", "rejected"];

    qualityElement.className = "calibration-quality calibration-quality--unknown";

    if (!supportedQualities.includes(quality)) {
        qualityElement.innerText = "Unavailable";
        reasonsElement.innerText = "No frame-quality result received";
        return;
    }

    qualityElement.innerText = quality.toUpperCase();
    qualityElement.className = "calibration-quality calibration-quality--" + quality;

    var reasons = Array.isArray(data.calibration_frame_reasons)
        ? data.calibration_frame_reasons
        : [];
    reasonsElement.innerText = reasons.length > 0
        ? reasons.map(formatCalibrationReason).join(", ")
        : "No quality issues detected";

    var awakeElement = document.getElementById("calibration_awake");
    var awakeReasonsElement = document.getElementById("calibration_awake_reasons");
    var evidenceElement = document.getElementById("calibration_evidence");
    var awakeEligible = data.calibration_awake_eligible === true;
    var awakeReasons = Array.isArray(data.calibration_awake_reasons)
        ? data.calibration_awake_reasons
        : [];

    awakeElement.innerText = awakeEligible ? "ELIGIBLE" : "WAITING";
    awakeElement.className = "calibration-quality " + (
        awakeEligible
            ? "calibration-quality--valid"
            : "calibration-quality--degraded"
    );
    awakeReasonsElement.innerText = awakeReasons.length > 0
        ? awakeReasons.map(formatCalibrationReason).join(", ")
        : "Recent observations are consistently awake";
    evidenceElement.innerText =
        (data.calibration_evidence_seconds ?? 0) + "s weighted evidence / " +
        (data.calibration_history_seconds ?? 0) + "s history";
}

function updateDashboard(data) {
    document.getElementById("ear_value").innerText = data.EAR;
    document.getElementById("mar_value").innerText = data.MAR;
    document.getElementById("drowsy_value").innerText =
        "Drowsiness Status: " + (data.is_drowsy ? "Yes 💤" : "No ✅");
    document.getElementById("ai_response").innerText = data.ai_response;
    updateCalibrationDiagnostic(data);
}

socket.on("update_data", function(data) {
    updateDashboard(data);
});

function requestData() {
    fetch('/data')
        .then(response => response.json())
        .then(data => {
            updateDashboard(data);
        })
        .catch(error => console.error('Error fetching data:', error));
}

requestData();
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
