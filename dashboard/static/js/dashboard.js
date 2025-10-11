const COLORS = {
    primary: '#E63946',
    secondary: '#1D3557',
    tertiary: '#457B9D',
    background: '#F8F9FA',
    text: '#1D1D1F'
};

const CHART_CONFIG = {
    responsive: true,
    maintainAspectRatio: true,
    aspectRatio: 2,
    plugins: {
        legend: {
            display: true,
            position: 'bottom',
            labels: {
                font: { family: 'Space Grotesk', size: 12, weight: '500' },
                color: '#6E6E73',
                padding: 16,
                usePointStyle: true,
                pointStyle: 'rect'
            }
        }
    },
    scales: {
        x: {
            grid: { display: false, drawBorder: false },
            ticks: {
                font: { family: 'Space Grotesk', size: 11 },
                color: '#86868B'
            }
        },
        y: {
            grid: { color: '#E1E4E8', drawBorder: false },
            ticks: {
                font: { family: 'Space Grotesk', size: 11 },
                color: '#86868B'
            }
        }
    }
};

let charts = {};
let currentData = { cournot: null, bertrand: null };

function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const views = document.querySelectorAll('.view-container');
    
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const viewName = item.getAttribute('data-view');
            
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');
            
            views.forEach(v => v.classList.add('hidden'));
            document.getElementById(`${viewName}-view`).classList.remove('hidden');
            
            const titles = {
                'overview': 'Market Analysis',
                'cournot': 'Cournot Competition',
                'bertrand': 'Bertrand Competition',
                'metrics': 'Performance Metrics'
            };
            document.querySelector('.page-title').textContent = titles[viewName] || 'Dashboard';
        });
    });
}

function initCharts() {
    const profitCtx = document.getElementById('profit-chart').getContext('2d');
    charts.profit = new Chart(profitCtx, {
        type: 'bar',
        data: {
            labels: ['Firm 1', 'Firm 2', 'Firm 3'],
            datasets: [{
                label: 'Nash Profit',
                data: [0, 0, 0],
                backgroundColor: [COLORS.primary, COLORS.secondary, COLORS.tertiary]
            }]
        },
        options: {
            ...CHART_CONFIG,
            aspectRatio: 1.8,
            plugins: {
                ...CHART_CONFIG.plugins,
                legend: { display: false }
            }
        }
    });
    
    const shareCtx = document.getElementById('share-chart').getContext('2d');
    charts.share = new Chart(shareCtx, {
        type: 'doughnut',
        data: {
            labels: ['Firm 1', 'Firm 2', 'Firm 3'],
            datasets: [{
                data: [33.3, 33.3, 33.3],
                backgroundColor: [COLORS.primary, COLORS.secondary, COLORS.tertiary],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 1.8,
            cutout: '70%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        font: { family: 'Space Grotesk', size: 12, weight: '500' },
                        color: '#6E6E73',
                        padding: 16,
                        usePointStyle: true,
                        pointStyle: 'rect'
                    }
                }
            }
        }
    });
    
    const cournotCtx = document.getElementById('cournot-chart').getContext('2d');
    charts.cournot = new Chart(cournotCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: []
        },
        options: CHART_CONFIG
    });
    
    const bertrandCtx = document.getElementById('bertrand-chart').getContext('2d');
    charts.bertrand = new Chart(bertrandCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: []
        },
        options: CHART_CONFIG
    });
}

function updateTimeSeriesChart(chartName, data, seriesType) {
    const chart = charts[chartName];
    if (!chart) return;
    
    const datasets = [];
    
    if (seriesType === 'prices' && chartName === 'cournot') {
        datasets.push({
            label: 'Market Price',
            data: data[seriesType],
            borderColor: COLORS.primary,
            backgroundColor: COLORS.primary,
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 4,
            tension: 0
        });
    } else {
        const firmCount = data[seriesType][0].length;
        
        for (let i = 0; i < firmCount; i++) {
            const colors = [COLORS.primary, COLORS.secondary, COLORS.tertiary];
            datasets.push({
                label: `Firm ${i + 1}`,
                data: data[seriesType].map(values => values[i]),
                borderColor: colors[i],
                backgroundColor: colors[i],
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 4,
                tension: 0
            });
        }
    }
    
    chart.data.labels = data.rounds;
    chart.data.datasets = datasets;
    chart.update();
}

function loadMetrics() {
    fetch('/api/metrics')
        .then(response => response.json())
        .then(data => {
            document.getElementById('hhi-value').textContent = data.hhi.toFixed(1);
            document.getElementById('nash-price-value').textContent = data.nash_price.toFixed(2);
            document.getElementById('surplus-value').textContent = data.total_surplus.toFixed(2);
            
            charts.profit.data.datasets[0].data = data.nash_profits;
            charts.profit.update();
            
            const totalQ = data.nash_quantities.reduce((a, b) => a + b, 0);
            const shares = data.nash_quantities.map(q => (q / totalQ * 100));
            charts.share.data.datasets[0].data = shares;
            charts.share.update();
            
            const tbody = document.getElementById('metrics-table-body');
            tbody.innerHTML = data.nash_quantities.map((q, i) => `
                <tr>
                    <td>Firm ${i + 1}</td>
                    <td>${q.toFixed(2)}</td>
                    <td>${data.nash_profits[i].toFixed(2)}</td>
                    <td>${shares[i].toFixed(1)}%</td>
                </tr>
            `).join('');
        });
}

function loadSimulation(market) {
    fetch(`/api/simulation/${market}`)
        .then(response => response.json())
        .then(data => {
            currentData[market] = data;
            const seriesType = market === 'cournot' ? 'quantities' : 'prices';
            updateTimeSeriesChart(market, data, seriesType);
        });
}

function initToggleButtons() {
    document.querySelectorAll('.toggle-group').forEach(group => {
        const buttons = group.querySelectorAll('.toggle-btn');
        buttons.forEach(btn => {
            btn.addEventListener('click', () => {
                buttons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                const series = btn.getAttribute('data-series');
                const view = group.closest('[id$="-view"]').id.replace('-view', '');
                
                if (currentData[view]) {
                    updateTimeSeriesChart(view, currentData[view], series);
                }
            });
        });
    });
}

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initCharts();
    initToggleButtons();
    loadMetrics();
    loadSimulation('cournot');
    loadSimulation('bertrand');
    
    document.getElementById('run-simulation-btn').addEventListener('click', (e) => {
        const btn = e.currentTarget;
        btn.disabled = true;
        btn.textContent = 'Running...';
        
        Promise.all([
            fetch('/api/metrics').then(r => r.json()),
            fetch('/api/simulation/cournot').then(r => r.json()),
            fetch('/api/simulation/bertrand').then(r => r.json())
        ]).then(([metricsData, cournotData, bertrandData]) => {
            // Update metrics (Overview tab)
            document.getElementById('hhi-value').textContent = metricsData.hhi.toFixed(1);
            document.getElementById('nash-price-value').textContent = metricsData.nash_price.toFixed(2);
            document.getElementById('surplus-value').textContent = metricsData.total_surplus.toFixed(2);
            charts.profit.data.datasets[0].data = metricsData.nash_profits;
            charts.profit.update();
            const totalQ = metricsData.nash_quantities.reduce((a, b) => a + b, 0);
            const shares = metricsData.nash_quantities.map(q => (q / totalQ * 100));
            charts.share.data.datasets[0].data = shares;
            charts.share.update();
            const tbody = document.getElementById('metrics-table-body');
            tbody.innerHTML = metricsData.nash_quantities.map((q, i) => `
                <tr>
                    <td>Firm ${i + 1}</td>
                    <td>${q.toFixed(2)}</td>
                    <td>${metricsData.nash_profits[i].toFixed(2)}</td>
                    <td>${shares[i].toFixed(1)}%</td>
                </tr>
            `).join('');
            
            // Update simulations
            currentData.cournot = cournotData;
            currentData.bertrand = bertrandData;
            updateTimeSeriesChart('cournot', cournotData, 'quantities');
            updateTimeSeriesChart('bertrand', bertrandData, 'prices');
            
            btn.disabled = false;
            btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M14 2V6H10" stroke="currentColor" stroke-width="2"/><path d="M2 14V10H6" stroke="currentColor" stroke-width="2"/><path d="M14 6C13.5 4 12 2.5 10 2C6 1 2 3 2 8C2 13 6 15 10 14C12 13.5 13.5 12 14 10" stroke="currentColor" stroke-width="2"/></svg> Run Simulation';
        });
    });
});

