document.addEventListener('DOMContentLoaded', () => {
    const uploadBtn = document.getElementById('uploadBtn');
    const imageInput = document.getElementById('imageInput');
    const dropZone = document.getElementById('dropZone');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const imagePlaceholder = document.getElementById('imagePlaceholder');
    const visionViewport = document.getElementById('visionViewport');
    const visionControls = document.getElementById('visionControls');
    const visionCanvas = document.getElementById('visionCanvas');
    const ctx = visionCanvas.getContext('2d');

    const origBtn = document.getElementById('origBtn');
    const maskBtn = document.getElementById('maskBtn');

    let currentData = null;
    let charts = {};

    // --- EVENT LISTENERS ---
    uploadBtn.addEventListener('click', () => imageInput.click());
    imageInput.addEventListener('change', (e) => handleFiles(e.target.files));

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        handleFiles(e.dataTransfer.files);
    });

    origBtn.addEventListener('click', () => {
        origBtn.classList.add('active');
        maskBtn.classList.remove('active');
        renderVision(false);
    });

    maskBtn.addEventListener('click', () => {
        maskBtn.classList.add('active');
        origBtn.classList.remove('active');
        renderVision(true);
    });

    // --- HANDLERS ---
    async function handleFiles(files) {
        if (files.length === 0) return;
        const file = files[0];
        if (!file.type.startsWith('image/')) return;

        const formData = new FormData();
        formData.append('file', file);

        loadingOverlay.classList.remove('hidden');

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            if (result.success) {
                currentData = result;
                updateDashboard(result);
            } else {
                alert('Analysis failed: ' + result.error);
            }
        } catch (err) {
            console.error(err);
            alert('Error connecting to server.');
        } finally {
            loadingOverlay.classList.add('hidden');
        }
    }

    function updateDashboard(data) {
        // Show Vision
        imagePlaceholder.classList.add('hidden');
        visionViewport.classList.remove('hidden');
        visionControls.classList.remove('hidden');

        // Update Text
        document.getElementById('riskValue').textContent = `${data.metrics.risk}%`;
        document.getElementById('severityLabel').textContent = data.metrics.severity;
        document.getElementById('infectionLabel').textContent = data.metrics.infection;
        
        document.getElementById('metricArea').textContent = `${data.metrics.area} px`;
        document.getElementById('metricHealth').textContent = `${data.metrics.healthy_pct}%`;
        document.getElementById('metricConf').textContent = `${data.metrics.confidence}%`;
        document.getElementById('metricRed').textContent = data.metrics.redness;

        // AI Explanation
        const formattedExp = data.explanation
            .replace(/SUMMARY:/g, '<b>📋 SUMMARY:</b>')
            .replace(/MEANING:/g, '<br><br><b>💡 MEANING:</b>')
            .replace(/RECOMMENDATION:/g, '<br><br><b>✅ RECOMMENDATION:</b>');
        document.getElementById('explanationContent').innerHTML = formattedExp;

        // Health Bar
        const hb = document.getElementById('healthBar');
        const score = data.metrics.health_score;
        hb.style.width = `${score}%`;
        hb.style.backgroundColor = score > 70 ? '#10b981' : score > 40 ? '#f59e0b' : '#f87171';

        // Render Image
        renderVision(maskBtn.classList.contains('active'));

        // Initialize Charts
        initCharts(data);
    }

    function renderVision(showMask) {
        if (!currentData) return;

        const img = new Image();
        img.onload = () => {
            visionCanvas.width = img.width;
            visionCanvas.height = img.height;
            visionCanvas.style.width = '100%';
            visionCanvas.style.height = 'auto';
            
            ctx.clearRect(0, 0, visionCanvas.width, visionCanvas.height);
            ctx.drawImage(img, 0, 0);
        };
        
        // Use the pre-rendered overlay if mask is requested, or clean original
        if (showMask && currentData.overlay) {
            img.src = 'data:image/jpeg;base64,' + currentData.overlay;
        } else {
            img.src = 'data:image/jpeg;base64,' + currentData.image;
        }
    }

    function initCharts(data) {
        // Destroy existing
        Object.values(charts).forEach(c => c.destroy());

        // 1. Metrics Bar
        charts.metrics = new Chart(document.getElementById('metricsChart'), {
            type: 'bar',
            data: {
                labels: ['Area', 'Redness', 'Irregularity'],
                datasets: [{
                    label: 'Normalized Value',
                    data: [data.metrics.norm.area, data.metrics.norm.red, data.metrics.norm.irreg],
                    backgroundColor: ['#f87171', '#f59e0b', '#38bdf8']
                }]
            },
            options: chartOptions('Metric Normalization', false)
        });

        // 2. Accuracy
        charts.acc = new Chart(document.getElementById('accChart'), {
            type: 'line',
            data: {
                labels: data.performance.epochs,
                datasets: [{
                    label: 'Validation Accuracy',
                    data: data.performance.accuracy,
                    borderColor: '#38bdf8',
                    tension: 0.4
                }]
            },
            options: chartOptions('Accuracy Evolution', true)
        });

        // 3. Loss
        charts.loss = new Chart(document.getElementById('lossChart'), {
            type: 'line',
            data: {
                labels: data.performance.epochs,
                datasets: [{
                    label: 'Training Loss',
                    data: data.performance.loss,
                    borderColor: '#f87171',
                    tension: 0.4
                }]
            },
            options: chartOptions('Model Convergence', true)
        });

        // 4. CM (Simplified Heatmap)
        charts.cm = new Chart(document.getElementById('cmChart'), {
            type: 'doughnut',
            data: {
                labels: ['Correct (Wound)', 'Incorrect', 'Correct (Healthy)'],
                datasets: [{
                    data: [45, 10, 45],
                    backgroundColor: ['#10b981', '#f87171', '#38bdf8']
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false }, title: { display: true, text: 'Classification Accuracy', color: 'white' } }
            }
        });
    }

    function chartOptions(title, showGrid) {
        return {
            responsive: true,
            plugins: {
                legend: { display: false },
                title: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: showGrid ? '#2d334a' : 'transparent' },
                    ticks: { color: '#94a3b8' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8' }
                }
            }
        };
    }
});
