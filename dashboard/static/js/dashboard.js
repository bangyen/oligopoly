const COLORS = {
    primary: '#E63946',
    secondary: '#1D3557',
    tertiary: '#457B9D',
    background: '#F8F9FA',
    text: '#1D1D1F'
};

function getThemeColors() {
    const isDark = document.body.getAttribute('data-theme') === 'dark';
    return {
        text: isDark ? '#8B949E' : '#6E6E73',
        grid: isDark ? '#30363D' : '#E1E4E8',
        textMain: isDark ? '#E6EDF3' : '#1D1D1F'
    };
}

function getChartConfig() {
    const theme = getThemeColors();
    return {
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 2,
        plugins: {
            legend: {
                display: true,
                position: 'bottom',
                labels: {
                    font: { family: 'Space Grotesk', size: 12, weight: '500' },
                    color: theme.text,
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
                    color: theme.text
                }
            },
            y: {
                grid: { color: theme.grid, drawBorder: false },
                ticks: {
                    font: { family: 'Space Grotesk', size: 11 },
                    color: theme.text
                }
            }
        }
    };
}

let charts = {};
let currentData = { cournot: null, bertrand: null };

let sessionStats = {
    runs: 0,
    cumulative: {
        quantities: [0, 0, 0],
        profits: [0, 0, 0],
        price: 0,
        surplus: 0
    }
};

function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item[data-view]');
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

function initTheme() {
    const themeToggle = document.getElementById('theme-toggle');
    const sunIcon = themeToggle.querySelector('.sun-icon');
    const moonIcon = themeToggle.querySelector('.moon-icon');

    function setTheme(isDark) {
        if (isDark) {
            document.body.setAttribute('data-theme', 'dark');
            sunIcon.style.display = 'none';
            moonIcon.style.display = 'block';
        } else {
            document.body.removeAttribute('data-theme');
            sunIcon.style.display = 'block';
            moonIcon.style.display = 'none';
        }
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
        updateChartsTheme();
    }

    // Check saved preference or system preference
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

    if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
        setTheme(true);
    }

    themeToggle.addEventListener('click', () => {
        const isDark = document.body.getAttribute('data-theme') === 'dark';
        setTheme(!isDark);
    });
}

function updateChartsTheme() {
    const theme = getThemeColors();
    const config = getChartConfig(); // Get fresh config with new colors

    Object.values(charts).forEach(chart => {
        if (!chart) return;

        // Update scales
        if (chart.options.scales.x) {
            chart.options.scales.x.ticks.color = theme.text;
        }
        if (chart.options.scales.y) {
            chart.options.scales.y.grid.color = theme.grid;
            chart.options.scales.y.ticks.color = theme.text;
        }

        // Update legend
        if (chart.options.plugins.legend) {
            chart.options.plugins.legend.labels.color = theme.text;
        }

        chart.update();
    });
}

function initCharts() {
    const config = getChartConfig();

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
            ...config,
            aspectRatio: 1.8,
            plugins: {
                ...config.plugins,
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
                        color: config.plugins.legend.labels.color,
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
        options: config
    });

    const bertrandCtx = document.getElementById('bertrand-chart').getContext('2d');
    charts.bertrand = new Chart(bertrandCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: []
        },
        options: config
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
            // Cyclical colors if more than 3 firms
            const colorList = [COLORS.primary, COLORS.secondary, COLORS.tertiary, '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51'];
            const color = colorList[i % colorList.length];

            datasets.push({
                label: `Firm ${i + 1}`,
                data: data[seriesType].map(values => values[i]),
                borderColor: color,
                backgroundColor: color,
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

function getSimulationParams() {
    const a = document.getElementById('config-a').value;
    const b = document.getElementById('config-b').value;
    const costs = document.getElementById('config-costs').value;

    const params = new URLSearchParams({
        a: a,
        b: b,
        alpha: parseFloat(a) * 2, // Heuristic: alpha usually larger than a for Bertrand in this demo context, or just use same? 
        // Actually, let's keep alpha independent or derive it. For simplicity, let's pass 'a' as 'alpha' if user wants, 
        // but the backend defaults are different (100 vs 200). 
        // Let's use the input 'a' for both a (Cournot) and alpha (Bertrand) but scale alpha if needed?
        // To keep it simple: we use 'a' input for both Intercepts.
        // Wait, the backend expects 'a' for Cournot and 'alpha' for Bertrand.
        costs: costs
    });

    // For Bertrand, we'll map 'a' to 'alpha' and 'b' to 'beta'
    // But typically Bertrand alpha is higher. Let's just pass them as is and let the user decide.
    // Or we can add specific inputs. For this 'MVP', let's use the inputs for both.
    params.append('alpha', a);
    params.append('beta', b);

    return params.toString();
}

function loadMetrics() {
    const query = getSimulationParams();
    Promise.all([
        fetch(`/api/metrics?${query}`).then(r => r.json()),
        fetch(`/api/simulation/cournot?${query}`).then(r => r.json())
    ]).then(([theoretical, cournotData]) => {
        const summary = cournotData.summary;

        // Reset or Update Session Stats Logic needs to handle changing N firms...
        // If N firms changes, cumulative stats might be invalid.
        // For simplicity, we just add up totals.

        // Count initial load as a run
        sessionStats.runs++;

        // Resize cumulative arrays if needed
        const numFirms = summary.avg_quantities.length;
        if (sessionStats.cumulative.quantities.length !== numFirms) {
            sessionStats.cumulative.quantities = new Array(numFirms).fill(0);
            sessionStats.cumulative.profits = new Array(numFirms).fill(0);
            // Reset session if firm count changes to avoid mixing data
            sessionStats.runs = 1;
        }

        for (let i = 0; i < numFirms; i++) {
            sessionStats.cumulative.quantities[i] += summary.avg_quantities[i];
            sessionStats.cumulative.profits[i] += summary.avg_profits[i];
        }
        sessionStats.cumulative.price += summary.avg_price;
        sessionStats.cumulative.surplus += summary.total_surplus;

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

        // Update Profit Chart
        charts.profit.data.labels = summary.avg_profits.map((_, i) => `Firm ${i + 1}`);
        charts.profit.data.datasets[0].data = summary.avg_profits;
        // Update colors if N firms > 3
        const colorList = [COLORS.primary, COLORS.secondary, COLORS.tertiary, '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51'];
        charts.profit.data.datasets[0].backgroundColor = summary.avg_profits.map((_, i) => colorList[i % colorList.length]);
        charts.profit.update();

        // Update Share Chart
        charts.share.data.labels = summary.avg_quantities.map((_, i) => `Firm ${i + 1}`);
        const totalQ = summary.avg_quantities.reduce((a, b) => a + b, 0);
        const shares = summary.avg_quantities.map(q => (q / totalQ * 100));
        charts.share.data.datasets[0].data = shares;
        charts.share.data.datasets[0].backgroundColor = summary.avg_quantities.map((_, i) => colorList[i % colorList.length]);
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

        // Update session count display
        document.getElementById('session-count').textContent =
            `Session: ${sessionStats.runs} run${sessionStats.runs !== 1 ? 's' : ''}`;

        // Update Cournot chart
        currentData.cournot = cournotData;
        updateTimeSeriesChart('cournot', cournotData, 'quantities');
    });
}

function loadSimulation(market) {
    const query = getSimulationParams();
    fetch(`/api/simulation/${market}?${query}`)
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
    initTheme();
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

        // Reload everything with new params
        Promise.all([
            fetch(`/api/metrics?${getSimulationParams()}`).then(r => r.json()),
            fetch(`/api/simulation/cournot?${getSimulationParams()}`).then(r => r.json()),
            fetch(`/api/simulation/bertrand?${getSimulationParams()}`).then(r => r.json())
        ]).then(([metricsData, cournotData, bertrandData]) => {

            // Note: Reuse logic from loadMetrics but adapted for the Run button flow
            // Actually, we can just call loadMetrics() and loadSimulation(), 
            // but the Run button logic in original code had session stats accumulation inline.
            // Let's refactor: simpler to just re-run loadMetrics which handles everything for Cournot/Summary
            // and loadSimulation('bertrand') for Bertrand.

            // However, the original code had specific accumulation logic. 
            // My updated loadMetrics handles accumulation too.
            // So calling them sequentially is fine, EXCEPT we need to wait for them to finish to re-enable button.

            // Let's just use the promise chain we started here to be safe and clean.

            const summary = cournotData.summary;
            const theoretical = metricsData;

            // Session accumulation logic duplicated from loadMetrics to be explicit? 
            // Or better: update loadMetrics to be reusable and return a promise.
            // For now, let's keep the logic here as it updates the UI immediately.

            sessionStats.runs++;
            const numFirms = summary.avg_quantities.length;

            // Reset if firm count changed
            if (sessionStats.cumulative.quantities.length !== numFirms) {
                sessionStats.cumulative.quantities = new Array(numFirms).fill(0);
                sessionStats.cumulative.profits = new Array(numFirms).fill(0);
                sessionStats.runs = 1;
            }

            for (let i = 0; i < numFirms; i++) {
                sessionStats.cumulative.quantities[i] += summary.avg_quantities[i];
                sessionStats.cumulative.profits[i] += summary.avg_profits[i];
            }
            sessionStats.cumulative.price += summary.avg_price;
            sessionStats.cumulative.surplus += summary.total_surplus;

            const sessionAvg = {
                quantities: sessionStats.cumulative.quantities.map(q => q / sessionStats.runs),
                profits: sessionStats.cumulative.profits.map(p => p / sessionStats.runs),
                price: sessionStats.cumulative.price / sessionStats.runs,
                surplus: sessionStats.cumulative.surplus / sessionStats.runs
            };

            document.getElementById('hhi-value').textContent = summary.hhi.toFixed(1);
            document.getElementById('nash-price-value').textContent = summary.avg_price.toFixed(2);
            document.getElementById('surplus-value').textContent = summary.total_surplus.toFixed(2);

            const priceDeviation = ((sessionAvg.price - theoretical.nash_price) / theoretical.nash_price * 100);
            const surplusDeviation = ((sessionAvg.surplus - theoretical.total_surplus) / theoretical.total_surplus * 100);

            document.getElementById('price-comparison').textContent =
                `${priceDeviation > 0 ? '+' : ''}${priceDeviation.toFixed(1)}% vs Nash`;
            document.getElementById('price-comparison').className =
                priceDeviation > 5 ? 'metric-change' : priceDeviation < -5 ? 'metric-change positive' : 'metric-change neutral';

            document.getElementById('surplus-comparison').textContent =
                `${surplusDeviation > 0 ? '+' : ''}${surplusDeviation.toFixed(1)}% vs Nash`;
            document.getElementById('surplus-comparison').className =
                surplusDeviation > 0 ? 'metric-change positive' : 'metric-change';

            // Update Profit Chart
            charts.profit.data.labels = summary.avg_profits.map((_, i) => `Firm ${i + 1}`);
            charts.profit.data.datasets[0].data = summary.avg_profits;
            const colorList = [COLORS.primary, COLORS.secondary, COLORS.tertiary, '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51'];
            charts.profit.data.datasets[0].backgroundColor = summary.avg_profits.map((_, i) => colorList[i % colorList.length]);
            charts.profit.update();

            // Update Share Chart
            charts.share.data.labels = summary.avg_quantities.map((_, i) => `Firm ${i + 1}`);
            const totalQ = summary.avg_quantities.reduce((a, b) => a + b, 0);
            const shares = summary.avg_quantities.map(q => (q / totalQ * 100));
            charts.share.data.datasets[0].data = shares;
            charts.share.data.datasets[0].backgroundColor = summary.avg_quantities.map((_, i) => colorList[i % colorList.length]);
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

            const comparisonTbody = document.getElementById('comparison-table-body');
            const rows = [
                {
                    metric: 'Market Price',
                    theoretical: theoretical.nash_price.toFixed(2),
                    actual: sessionAvg.price.toFixed(2),
                    deviation: priceDeviation
                },
                {
                    metric: 'Total Surplus',
                    theoretical: theoretical.total_surplus.toFixed(2),
                    actual: sessionAvg.surplus.toFixed(2),
                    deviation: surplusDeviation
                },
                ...sessionAvg.quantities.map((q, i) => ({
                    metric: `Firm ${i + 1} Quantity`,
                    theoretical: theoretical.nash_quantities[i].toFixed(2),
                    actual: q.toFixed(2),
                    deviation: ((q - theoretical.nash_quantities[i]) / theoretical.nash_quantities[i] * 100)
                })),
                ...sessionAvg.profits.map((p, i) => ({
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

            document.getElementById('session-count').textContent =
                `Session: ${sessionStats.runs} run${sessionStats.runs !== 1 ? 's' : ''}`;

            currentData.cournot = cournotData;
            currentData.bertrand = bertrandData;
            updateTimeSeriesChart('cournot', cournotData, 'quantities');
            updateTimeSeriesChart('bertrand', bertrandData, 'prices');

            btn.disabled = false;
            btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M14 8C14 11.3137 11.3137 14 8 14C4.68629 14 2 11.3137 2 8C2 4.68629 4.68629 2 8 2C10.3 2 12.3 3.2 13.4 5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M13 2V5H10" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg> Run Simulation';
        });
    });

    document.getElementById('reset-session-btn').addEventListener('click', () => {
        sessionStats.runs = 0;
        sessionStats.cumulative = {
            quantities: [0, 0, 0],
            profits: [0, 0, 0],
            price: 0,
            surplus: 0
        };
        document.getElementById('session-count').textContent = 'Session: 0 runs';
        document.getElementById('comparison-table-body').innerHTML = '<tr><td colspan="4" class="loading">Run a simulation to see data...</td></tr>';
        document.getElementById('price-comparison').textContent = '—';
        document.getElementById('surplus-comparison').textContent = '—';
    });
});

