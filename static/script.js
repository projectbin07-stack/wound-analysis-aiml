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

        // Update Text. Without a trained model there is no score, band or confidence.
        const m = data.metrics;
        const na = 'n/a (no model)';
        document.getElementById('riskValue').textContent = m.scored ? `${m.risk}%` : 'n/a';
        document.getElementById('severityLabel').textContent = m.scored ? m.severity : na;
        document.getElementById('rednessBandLabel').textContent = m.scored ? m.redness_band : na;
        
        document.getElementById('metricArea').textContent = `${m.area} px`;
        document.getElementById('metricHealth').textContent = m.scored ? `${m.healthy_pct}%` : na;
        document.getElementById('metricConf').textContent = m.scored ? `${m.confidence}%` : na;
        document.getElementById('metricRed').textContent = m.redness;

        // AI Explanation
        // Escape first: the text may come from an LLM and must never be injected as HTML.
        const escaped = String(data.explanation)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        const formattedExp = escaped
            .replace(/SUMMARY:/g, '<b>SUMMARY:</b>')
            .replace(/MEANING:/g, '<br><br><b>MEANING:</b>')
            .replace(/NOTE:/g, '<br><br><b>NOTE:</b>');
        document.getElementById('explanationContent').innerHTML = formattedExp;

        // Health Bar
        const hb = document.getElementById('healthBar');
        const score = m.scored ? m.health_score : 0;
        hb.style.width = `${score}%`;
        hb.style.backgroundColor = score > 70 ? '#10b981' : score > 40 ? '#f59e0b' : '#f87171';

        // Tell the user which segmentation path produced this result
        const banner = document.getElementById('modeBanner');
        if (data.mode === 'model') {
            banner.classList.add('hidden');
        } else {
            banner.textContent = 'No model loaded: heuristic fallback. The outline comes from a simple brightness threshold, not from the neural network, and is not a wound detection. No score, band or confidence is reported.';
            banner.classList.remove('hidden');
        }

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
