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
    Promise.all([
        fetch('/api/metrics').then(r => r.json()),
        fetch('/api/simulation/cournot').then(r => r.json())
    ]).then(([theoretical, cournotData]) => {
        const summary = cournotData.summary;
        
        document.getElementById('hhi-value').textContent = summary.hhi.toFixed(1);
        document.getElementById('nash-price-value').textContent = summary.avg_price.toFixed(2);
        document.getElementById('surplus-value').textContent = summary.total_surplus.toFixed(2);
        
        // Update comparison labels
        const priceDeviation = ((summary.avg_price - theoretical.nash_price) / theoretical.nash_price * 100);
        const surplusDeviation = ((summary.total_surplus - theoretical.total_surplus) / theoretical.total_surplus * 100);
            
            document.getElementById('price-comparison').textContent = 
                `${priceDeviation > 0 ? '+' : ''}${priceDeviation.toFixed(1)}% vs Nash`;
            document.getElementById('price-comparison').className = 
                priceDeviation > 5 ? 'metric-change' : priceDeviation < -5 ? 'metric-change positive' : 'metric-change neutral';
                
        document.getElementById('surplus-comparison').textContent = 
            `${surplusDeviation > 0 ? '+' : ''}${surplusDeviation.toFixed(1)}% vs Nash`;
        document.getElementById('surplus-comparison').className = 
            surplusDeviation > 0 ? 'metric-change positive' : 'metric-change';
        
        charts.profit.data.datasets[0].data = summary.avg_profits;
        charts.profit.update();
        
        const totalQ = summary.avg_quantities.reduce((a, b) => a + b, 0);
        const shares = summary.avg_quantities.map(q => (q / totalQ * 100));
        charts.share.data.datasets[0].data = shares;
        charts.share.update();
        
        const tbody = document.getElementById('metrics-table-body');
        tbody.innerHTML = summary.avg_quantities.map((q, i) => `
            <tr>
                <td>Firm ${i + 1}</td>
                <td>${q.toFixed(2)}</td>
                <td>${summary.avg_profits[i].toFixed(2)}</td>
                <td>${shares[i].toFixed(1)}%</td>
            </tr>
        `).join('');
        
        // Populate comparison table
        const comparisonTbody = document.getElementById('comparison-table-body');
        const rows = [
            {
                metric: 'Market Price',
                theoretical: theoretical.nash_price.toFixed(2),
                actual: summary.avg_price.toFixed(2),
                deviation: priceDeviation
            },
            {
                metric: 'Total Surplus',
                theoretical: theoretical.total_surplus.toFixed(2),
                actual: summary.total_surplus.toFixed(2),
                deviation: surplusDeviation
            },
            ...summary.avg_quantities.map((q, i) => ({
                metric: `Firm ${i + 1} Quantity`,
                theoretical: theoretical.nash_quantities[i].toFixed(2),
                actual: q.toFixed(2),
                deviation: ((q - theoretical.nash_quantities[i]) / theoretical.nash_quantities[i] * 100)
            })),
            ...summary.avg_profits.map((p, i) => ({
                metric: `Firm ${i + 1} Profit`,
                theoretical: theoretical.nash_profits[i].toFixed(2),
                actual: p.toFixed(2),
                deviation: ((p - theoretical.nash_profits[i]) / theoretical.nash_profits[i] * 100)
            }))
        ];
        
        comparisonTbody.innerHTML = rows.map(row => `
            <tr>
                <td>${row.metric}</td>
                <td>${row.theoretical}</td>
                <td>${row.actual}</td>
                <td style="color: ${Math.abs(row.deviation) > 10 ? '#E63946' : Math.abs(row.deviation) > 5 ? '#457B9D' : '#06D6A0'}">${row.deviation > 0 ? '+' : ''}${row.deviation.toFixed(1)}%</td>
            </tr>
        `).join('');
        
        // Update Cournot chart
        currentData.cournot = cournotData;
        updateTimeSeriesChart('cournot', cournotData, 'quantities');
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
            // Use Cournot simulation summary for Overview (not a separate simulation)
            const summary = cournotData.summary;
            const theoretical = metricsData;
            
            document.getElementById('hhi-value').textContent = summary.hhi.toFixed(1);
            document.getElementById('nash-price-value').textContent = summary.avg_price.toFixed(2);
            document.getElementById('surplus-value').textContent = summary.total_surplus.toFixed(2);
            
            // Update comparison labels
            const priceDeviation = ((summary.avg_price - theoretical.nash_price) / theoretical.nash_price * 100);
            const surplusDeviation = ((summary.total_surplus - theoretical.total_surplus) / theoretical.total_surplus * 100);
            
            document.getElementById('price-comparison').textContent = 
                `${priceDeviation > 0 ? '+' : ''}${priceDeviation.toFixed(1)}% vs Nash`;
            document.getElementById('price-comparison').className = 
                priceDeviation > 5 ? 'metric-change' : priceDeviation < -5 ? 'metric-change positive' : 'metric-change neutral';
                
            document.getElementById('surplus-comparison').textContent = 
                `${surplusDeviation > 0 ? '+' : ''}${surplusDeviation.toFixed(1)}% vs Nash`;
            document.getElementById('surplus-comparison').className = 
                surplusDeviation > 0 ? 'metric-change positive' : 'metric-change';
            
            charts.profit.data.datasets[0].data = summary.avg_profits;
            charts.profit.update();
            const totalQ = summary.avg_quantities.reduce((a, b) => a + b, 0);
            const shares = summary.avg_quantities.map(q => (q / totalQ * 100));
            charts.share.data.datasets[0].data = shares;
            charts.share.update();
            const tbody = document.getElementById('metrics-table-body');
            tbody.innerHTML = summary.avg_quantities.map((q, i) => `
                <tr>
                    <td>Firm ${i + 1}</td>
                    <td>${q.toFixed(2)}</td>
                    <td>${summary.avg_profits[i].toFixed(2)}</td>
                    <td>${shares[i].toFixed(1)}%</td>
                </tr>
            `).join('');
            
            // Update comparison table
            const comparisonTbody = document.getElementById('comparison-table-body');
            const rows = [
                {
                    metric: 'Market Price',
                    theoretical: theoretical.nash_price.toFixed(2),
                    actual: summary.avg_price.toFixed(2),
                    deviation: priceDeviation
                },
                {
                    metric: 'Total Surplus',
                    theoretical: theoretical.total_surplus.toFixed(2),
                    actual: summary.total_surplus.toFixed(2),
                    deviation: surplusDeviation
                },
                ...summary.avg_quantities.map((q, i) => ({
                    metric: `Firm ${i + 1} Quantity`,
                    theoretical: theoretical.nash_quantities[i].toFixed(2),
                    actual: q.toFixed(2),
                    deviation: ((q - theoretical.nash_quantities[i]) / theoretical.nash_quantities[i] * 100)
                })),
                ...summary.avg_profits.map((p, i) => ({
                    metric: `Firm ${i + 1} Profit`,
                    theoretical: theoretical.nash_profits[i].toFixed(2),
                    actual: p.toFixed(2),
                    deviation: ((p - theoretical.nash_profits[i]) / theoretical.nash_profits[i] * 100)
                }))
            ];
            
            comparisonTbody.innerHTML = rows.map(row => `
                <tr>
                    <td>${row.metric}</td>
                    <td>${row.theoretical}</td>
                    <td>${row.actual}</td>
                    <td style="color: ${Math.abs(row.deviation) > 10 ? '#E63946' : Math.abs(row.deviation) > 5 ? '#457B9D' : '#06D6A0'}">${row.deviation > 0 ? '+' : ''}${row.deviation.toFixed(1)}%</td>
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

