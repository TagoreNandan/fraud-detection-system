/**
 * Fraud Detection System - Frontend Application Logic
 * 
 * This file handles all interactive elements of the application, broken down
 * by feature tab for better clarity and maintainability.
 */

document.addEventListener("DOMContentLoaded", () => {
    // Initialize all components once the DOM is ready
    initNavigation();
    initScorerTab();
    initSimulatorTab();
    initDashboardTab();
    initBatchUploadTab();
});


// ==========================================
// 1. Navigation Logic
// ==========================================
function initNavigation() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // 1. Hide all tabs and remove 'active' styling from all buttons
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.style.display = 'none');

            // 2. Show the clicked tab and add 'active' styling
            btn.classList.add('active');
            const targetId = btn.dataset.target;
            document.getElementById(targetId).style.display = 'block';

            // 3. Trigger specific logic if a certain tab is opened
            if (targetId === 'tab-dashboard') {
                // The dashboard needs to fetch fresh data every time it's opened
                window.loadDashboardData();
            }
        });
    });
}


// ==========================================
// 2. Transaction Scorer Tab (Manual Analysis)
// ==========================================
function initScorerTab() {
    setupAdvancedFeaturesToggle();
    setupSampleDataButtons();
    setupPredictionForm();
}

/**
 * Dynamically generates 28 input fields for the anonymized V1-V28 PCA features.
 * Also handles the "Show/Hide Advanced Features" accordion button.
 */
function setupAdvancedFeaturesToggle() {
    const vFeaturesGrid = document.getElementById("v-features-grid");
    const btnToggleV = document.getElementById("btn-toggle-v");
    const toggleIcon = document.getElementById("toggle-v-icon");
    const toggleText = document.getElementById("toggle-v-text");

    // Generate V1 to V28 input fields dynamically to keep HTML clean
    if (vFeaturesGrid) {
        for (let i = 1; i <= 28; i++) {
            const div = document.createElement("div");
            div.className = "input-group";

            div.innerHTML = `
                <label>V${i}</label>
                <input type="number" step="any" id="V${i}" name="V${i}" required>
            `;
            vFeaturesGrid.appendChild(div);
        }
    }

    // Toggle accordion logic
    if (btnToggleV) {
        btnToggleV.addEventListener("click", () => {
            const isHidden = vFeaturesGrid.style.display === "none";

            vFeaturesGrid.style.display = isHidden ? "grid" : "none";
            toggleIcon.className = isHidden ? "ph ph-caret-up" : "ph ph-caret-down";
            toggleText.innerText = isHidden ? "Hide Advanced Features" : "Show Advanced Features (V1-V28)";
            btnToggleV.style.background = isHidden ? "var(--bg-panel-hover)" : "transparent";
        });
    }
}

/**
 * Wires up the "Legit Sample" and "Fraud Sample" buttons to quickly
 * populate the form for testing purposes.
 */
function setupSampleDataButtons() {
    const btnLegit = document.getElementById("btn-sample-legit");
    const btnFraud = document.getElementById("btn-sample-fraud");

    // Pre-defined testing profiles
    const legitSample = {
        Time: 0.0, Amount: 149.62,
        V1: -1.359807, V2: -0.072781, V3: 2.536346, V4: 1.378155, V5: -0.338321,
        V6: 0.462388, V7: 0.239599, V8: 0.098698, V9: 0.363787, V10: 0.090794,
        V11: -0.551600, V12: -0.617801, V13: -0.991390, V14: -0.311169, V15: 1.468177,
        V16: -0.470401, V17: 0.207971, V18: 0.025791, V19: 0.403993, V20: 0.251412,
        V21: -0.018307, V22: 0.277838, V23: -0.110474, V24: 0.066928, V25: 0.128539,
        V26: -0.189115, V27: 0.133558, V28: -0.021053
    };

    const fraudSample = {
        Time: 406.0, Amount: 0.0,
        V1: -2.312227, V2: 1.951992, V3: -1.609851, V4: 3.997906, V5: -0.522188,
        V6: -1.426545, V7: -2.537387, V8: 1.391657, V9: -2.770089, V10: -2.772272,
        V11: 3.202033, V12: -2.899907, V13: -0.595222, V14: -4.289254, V15: 0.389724,
        V16: -1.140747, V17: -2.830056, V18: -0.016822, V19: 0.416956, V20: 0.126911,
        V21: 0.517232, V22: -0.035049, V23: -0.465211, V24: 0.320198, V25: 0.044519,
        V26: 0.177840, V27: 0.261145, V28: -0.143276
    };

    const fillForm = (data) => {
        Object.keys(data).forEach(key => {
            const input = document.getElementById(key);
            if (input) {
                input.value = data[key];
                // Flash the border to show it was populated
                input.style.borderColor = "var(--accent)";
                setTimeout(() => input.style.borderColor = "var(--border-color)", 500);
            }
        });
    };

    if (btnLegit) btnLegit.addEventListener("click", () => fillForm(legitSample));
    if (btnFraud) btnFraud.addEventListener("click", () => fillForm(fraudSample));
}

/**
 * Handles the main prediction form submission, sending data to the XGBoost API
 * and rendering the results (including SHAP values).
 */
function setupPredictionForm() {
    const form = document.getElementById("prediction-form");
    const resultsContent = document.getElementById("results-content");

    if (!form) return;

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        // 1. Show loading state
        resultsContent.innerHTML = `
            <div class="loader-container">
                <div class="spinner"></div>
                <h2>Analyzing Transaction...</h2>
                <p style="color: var(--text-secondary)">Calculating predictions and SHAP values</p>
            </div>
        `;

        // 2. Gather form data into a JSON object
        const formData = new FormData(form);
        const payload = {};
        formData.forEach((value, key) => { payload[key] = parseFloat(value); });

        // 3. Send to API
        const startTime = performance.now();
        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error("Prediction API failed");

            const result = await response.json();
            const latency = (performance.now() - startTime).toFixed(0);

            // 4. Render results
            renderPredictionResults(result, latency, resultsContent);
        } catch (error) {
            resultsContent.innerHTML = `
                <div class="empty-state">
                    <i class="ph ph-warning-circle" style="font-size: 3rem; color: var(--status-fraud)"></i>
                    <p>Connection Error. Is the backend API running?</p>
                </div>
            `;
        }
    });
}

/**
 * Helper function to draw the prediction result card and the SHAP explanation bars
 */
function renderPredictionResults(result, latency, container) {
    const isFraud = result.prediction === "Fraud";
    const probPct = (result.fraud_probability * 100).toFixed(2);

    // Generate SHAP explanation bars HTML if available
    let shapHtml = "";
    if (result.top_risk_factors?.length > 0) {
        const maxImpact = Math.max(...result.top_risk_factors.map(f => f.impact));

        shapHtml = result.top_risk_factors.map(f => {
            const widthPct = (f.impact / maxImpact) * 100;
            // Red gradient for fraud drivers, blue for legit drivers
            const barStyle = isFraud
                ? 'background: linear-gradient(90deg, #ef4444, #f87171);'
                : '';

            return `
                <div class="shap-item">
                    <div class="shap-label">
                        <span>${f.feature}</span>
                        <span>${f.impact.toFixed(4)}</span>
                    </div>
                    <div class="shap-bar-bg">
                        <div class="shap-bar-fill" style="width: ${widthPct}%; ${barStyle}"></div>
                    </div>
                </div>
            `;
        }).join("");
    }

    // Update container
    container.innerHTML = `
        <div class="result-card ${isFraud ? 'fraud' : 'legit'}">
            <i class="ph ${isFraud ? 'ph-shield-warning' : 'ph-shield-check'} result-icon"></i>
            <h2 class="result-title">${isFraud ? 'FRAUD DETECTED' : 'LEGITIMATE'}</h2>
            <p class="result-prob">Risk Probability: <span>${probPct}%</span></p>
            <p style="font-size: 0.8rem; margin-top: 0.5rem; opacity: 0.7;">Processed in ${latency}ms</p>
        </div>
        ${shapHtml ? `
        <div class="shap-section">
            <h3><i class="ph ph-target"></i> Key Risk Drivers (SHAP)</h3>
            <div class="shap-bars">${shapHtml}</div>
        </div>` : ""}
    `;
}


// ==========================================
// 3. Simulator Tab (Automated Testing)
// ==========================================
function initSimulatorTab() {
    let simInterval = null;
    const btnStartSim = document.getElementById("btn-start-sim");
    const btnStopSim = document.getElementById("btn-stop-sim");
    const simSpeed = document.getElementById("sim-speed");
    const simStatus = document.getElementById("sim-status");
    const simLogs = document.getElementById("sim-logs");

    if (!btnStartSim || !btnStopSim) return;

    // Start Simulation Button
    btnStartSim.addEventListener("click", () => {
        const delayMs = parseInt(simSpeed.value, 10);

        // Start the generation loop
        simInterval = setInterval(() => runSimulatorTick(simLogs), delayMs);

        // Update UI states
        btnStartSim.disabled = true;
        btnStartSim.style.opacity = '0.5';
        btnStopSim.disabled = false;
        btnStopSim.style.opacity = '1';
        simStatus.className = 'status-badge running';
        simStatus.textContent = 'Running';
    });

    // Stop Simulation Button
    btnStopSim.addEventListener("click", () => {
        // Clear the generation loop
        clearInterval(simInterval);
        simInterval = null;

        // Update UI states
        btnStartSim.disabled = false;
        btnStartSim.style.opacity = '1';
        btnStopSim.disabled = true;
        btnStopSim.style.opacity = '0.5';
        simStatus.className = 'status-badge stopped';
        simStatus.textContent = 'Stopped';
    });
}

/**
 * Generates a synthetic transaction and sends it to the API, logging the results.
 * Called on an interval by the simulator.
 */
async function runSimulatorTick(simLogsContainer) {
    // 1. Generate random synthetic features
    const amount = Math.random() < 0.1 ? (Math.random() * 195000 + 5000) : (Math.random() * 4990 + 10);
    const transaction = { Time: Math.random() * 172800, Amount: amount };
    for (let i = 1; i <= 28; i++) {
        transaction[`V${i}`] = Math.random() < 0.05 ? (Math.random() * 16 - 8) : (Math.random() * 4 - 2);
    }

    // 2. Send to API and log result
    const timeStr = new Date().toLocaleTimeString();
    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(transaction)
        });

        if (!res.ok) throw new Error("API Error");

        const data = await res.json();
        const isFraud = data.prediction === 'Fraud';

        // Build log HTML
        const logHtml = `<span>[${timeStr}] Amount: ₹${amount.toFixed(2)}</span><span>${data.prediction} (${data.fraud_probability.toFixed(2)})</span>`;
        appendSimulationLog(simLogsContainer, logHtml, isFraud);

    } catch (err) {
        // Build error log HTML
        const logHtml = `<span>[${timeStr}] Amount: ₹${amount.toFixed(2)}</span><span>Connection Error</span>`;
        appendSimulationLog(simLogsContainer, logHtml, true);
    }
}

/** Helper to keep the simulator log scrollable and limited to 50 items */
function appendSimulationLog(container, innerHTML, isAlert) {
    const div = document.createElement("div");
    div.className = `sim-log-item ${isAlert ? 'fraud-log' : 'legit-log'}`;
    div.innerHTML = innerHTML;

    container.prepend(div);
    if (container.children.length > 50) {
        container.removeChild(container.lastChild);
    }
}


// ==========================================
// 4. Dashboard Tab (Metrics & Charting)
// ==========================================
// Expose loadDashboard to window so the Navigation logic can call it
window.loadDashboardData = async () => {
    try {
        // 1. Fetch top-level KPIs
        const metricsRes = await fetch("/metrics");
        const metrics = await metricsRes.json();

        document.getElementById("dash-total").innerText = metrics.total_transactions || 0;
        document.getElementById("dash-frauds").innerText = metrics.frauds_detected || 0;
        document.getElementById("dash-rate").innerText = ((metrics.fraud_rate || 0) * 100).toFixed(2) + "%";

        // 2. Fetch the latest transactions for charting and tables
        const txRes = await fetch("/transactions?limit=100");
        const transactions = await txRes.json();

        renderDashboardTable(transactions);
        renderDashboardChart(transactions);

    } catch (err) {
        console.error("Dashboard fetch error:", err);
    }
};

function renderDashboardTable(transactions) {
    const tbody = document.querySelector("#recent-tx-table tbody");
    if (!tbody) return;

    tbody.innerHTML = transactions.map(t => {
        const isFraud = t.prediction === 'Fraud';
        const dateStr = new Date(t.timestamp).toLocaleTimeString();
        return `
            <tr>
                <td>${dateStr}</td>
                <td>₹${t.amount.toFixed(2)}</td>
                <td><span class="prob-badge ${isFraud ? 'prob-fraud' : 'prob-legit'}">${t.prediction}</span></td>
                <td>${(t.fraud_probability * 100).toFixed(2)}%</td>
            </tr>
        `;
    }).join('');
}

// Global variable to keep track of the Chart.js instance
let fraudChartInstance = null;

function renderDashboardChart(transactions) {
    const ctx = document.getElementById('fraudProbabilityChart')?.getContext('2d');
    if (!ctx) return;

    // Sort ascending for chart so time goes left to right
    const chartData = [...transactions].reverse();
    const labels = chartData.map(t => new Date(t.timestamp).toLocaleTimeString());
    const dataPts = chartData.map(t => t.fraud_probability);

    // Update existing chart or create a new one
    if (fraudChartInstance) {
        fraudChartInstance.data.labels = labels;
        fraudChartInstance.data.datasets[0].data = dataPts;
        fraudChartInstance.update();
    } else {
        fraudChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Fraud Probability',
                    data: dataPts,
                    borderColor: '#38bdf8',
                    backgroundColor: 'rgba(56, 189, 248, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, max: 1, grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: '#94a3b8' } },
                    x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8', maxTicksLimit: 10 } }
                },
                plugins: { legend: { labels: { color: '#f8fafc' } } }
            }
        });
    }
}

function initDashboardTab() {
    // Hook up action buttons in the dashboard
    const btnRefreshDash = document.getElementById("btn-refresh-dashboard");
    if (btnRefreshDash) btnRefreshDash.addEventListener("click", window.loadDashboardData);

    const btnResetDash = document.getElementById("btn-reset-db");
    const simLogs = document.getElementById("sim-logs");

    if (btnResetDash) {
        btnResetDash.addEventListener("click", async () => {
            if (confirm("Are you sure you want to reset all transactions to 0? This cannot be undone.")) {
                try {
                    const res = await fetch("/reset", { method: "POST" });
                    if (res.ok) {
                        alert("Database reset successfully.");
                        window.loadDashboardData();
                        if (simLogs) simLogs.innerHTML = "";
                    } else {
                        throw new Error("Reset failed");
                    }
                } catch (err) {
                    alert("Error resetting database: " + err.message);
                }
            }
        });
    }
}


// ==========================================
// 5. Batch Upload Tab (CSV Processing)
// ==========================================
function initBatchUploadTab() {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("csv-file-input");

    if (!dropZone || !fileInput) return;

    // Click to open file dialog
    dropZone.addEventListener("click", () => fileInput.click());

    // Drag and drop visual state handlers
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, e => { e.preventDefault(); e.stopPropagation(); }, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    // Handle incoming files
    dropZone.addEventListener('drop', e => processUploadedFiles(e.dataTransfer.files));
    fileInput.addEventListener('change', function () { processUploadedFiles(this.files); });
}

function processUploadedFiles(files) {
    if (!files || !files.length) return;

    const file = files[0];
    if (!file.name.endsWith('.csv')) {
        alert('Please upload a valid CSV file.');
        return;
    }

    // Read the file locally via FileReader API
    const reader = new FileReader();
    reader.onload = async (e) => {
        const csvText = e.target.result;
        await sendBatchCsvToApi(csvText);
    };
    reader.readAsText(file);
}

/**
 * Parses the CSV locally, builds a JSON payload, and sends it to the API 
 * for batch prediction.
 */
async function sendBatchCsvToApi(csvText) {
    const batchProgress = document.getElementById("batch-progress");
    const progressFill = document.getElementById("batch-progress-fill");
    const batchResults = document.getElementById("batch-results");

    // 1. Setup UI for processing
    batchProgress.style.display = 'block';
    batchResults.style.display = 'none';
    progressFill.style.width = '10%';

    // 2. Parse CSV text
    const rows = csvText.trim().split('\n').map(r => r.split(','));
    const headers = rows[0].map(h => h.trim());

    // Ensure we have all 30 expected columns (Time, Amount, V1-V28)
    const reqCols = ['Time', 'Amount'];
    for (let i = 1; i <= 28; i++) reqCols.push(`V${i}`);

    const indices = {};
    for (let col of reqCols) {
        const idx = headers.findIndex(h => h === col);
        if (idx === -1) {
            alert(`CSV is missing required column: ${col}`);
            batchProgress.style.display = 'none';
            return;
        }
        indices[col] = idx;
    }

    progressFill.style.width = '30%';

    // 3. Build JSON transactions array from CSV rows
    const transactions = [];
    for (let i = 1; i < rows.length; i++) {
        if (rows[i].length !== headers.length) continue; // Skip malformed rows

        const tx = {};
        for (let col of reqCols) {
            tx[col] = parseFloat(rows[i][indices[col]]);
        }

        if (!isNaN(tx.Amount)) {
            transactions.push(tx);
        }
    }

    if (transactions.length === 0) {
        alert('No valid transactions found in the CSV.');
        batchProgress.style.display = 'none';
        return;
    }

    progressFill.style.width = '60%';

    // 4. Send to backend batch endpoint
    try {
        const res = await fetch("/batch-predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ transactions })
        });

        if (!res.ok) throw new Error("Batch Predict Failed");
        const data = await res.json();

        progressFill.style.width = '100%';

        // Give the progress bar time to visually complete before rendering
        setTimeout(() => {
            batchProgress.style.display = 'none';
            renderBatchTable(transactions, data.results);
        }, 500);

    } catch (err) {
        alert('Error calling batch API: ' + err.message);
        batchProgress.style.display = 'none';
    }
}

/**
 * Renders the results of the batch upload into a summary and table
 */
function renderBatchTable(inputs, results) {
    document.getElementById("batch-results").style.display = 'block';
    document.getElementById("batch-total").innerText = inputs.length;

    let frauds = 0;
    const tbody = document.querySelector("#batch-results-table tbody");
    let html = '';

    results.forEach((r, i) => {
        if (r.prediction === 'Fraud') frauds++;

        // Only render the first 100 to prevent browser freezing on large CSVs
        if (i < 100) {
            const isFraud = r.prediction === 'Fraud';
            html += `
                <tr>
                    <td>${i + 1}</td>
                    <td>₹${inputs[i].Amount.toFixed(2)}</td>
                    <td><span class="prob-badge ${isFraud ? 'prob-fraud' : 'prob-legit'}">${r.prediction}</span></td>
                    <td>${(r.fraud_probability * 100).toFixed(2)}%</td>
                </tr>
            `;
        }
    });

    if (inputs.length > 100) {
        html += `<tr><td colspan="4" style="text-align: center; color: var(--text-secondary);">Showing first 100 results (${inputs.length - 100} more hidden)...</td></tr>`;
    }

    document.getElementById("batch-frauds").innerText = frauds;
    tbody.innerHTML = html;
}

