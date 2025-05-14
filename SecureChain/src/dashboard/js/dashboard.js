// Sample code to fetch and display event logs dynamically
document.addEventListener('DOMContentLoaded', function() {
    fetch('/api/events')  // Example API endpoint to get events
        .then(response => response.json())
        .then(data => {
            const table = document.getElementById('event-log').getElementsByTagName('tbody')[0];
            data.forEach(event => {
                const row = table.insertRow();
                row.insertCell(0).textContent = event.eventId;
                row.insertCell(1).textContent = new Date(event.timestamp * 1000).toLocaleString();
                row.insertCell(2).textContent = event.eventType;
                row.insertCell(3).textContent = event.status ? "Anomaly Detected" : "Normal";
            });
        })
        .catch(console.error);
});
