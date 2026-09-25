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

    // Scenario series colours depend on the theme
    if (typeof scenarioResult !== 'undefined' && scenarioResult) renderScenario();
}

// ---------------------------------------------------------------------------
// Scenario Lab: runs the full engine (POST /api/scenario) with any demand
// system, per-firm strategy and market evolution.
// ---------------------------------------------------------------------------

const STRATEGIES = [
    ['nash', 'Adaptive Nash'],
    ['fictitious_play', 'Fictitious play'],
    ['q_learning', 'Q-learning'],
    ['deep_q_learning', 'Deep Q-learning'],
    ['behavioral', 'Behavioral'],
    ['cartel', 'Cartel (loyal)'],
    ['collusive', 'Collusive (may defect)'],
    ['opportunistic', 'Opportunistic']
];

const SERIES_COLORS = ['#E63946', '#1D3557', '#457B9D', '#F4A261', '#2A9D8F', '#8D99AE', '#6D597A', '#B56576'];

let scenarioResult = null;
let scenarioSeries = 'actions';

function parseCsv(text) {
    return text.split(',').map(v => v.trim()).filter(Boolean).map(Number);
}

function renderStrategyPickers() {
    const container = document.getElementById('sc-strategies');
    const costs = parseCsv(document.getElementById('sc-costs').value);
    const previous = Array.from(container.querySelectorAll('select')).map(s => s.value);
    container.innerHTML = '';
    costs.forEach((cost, i) => {
        const label = document.createElement('label');
        label.className = 'config-label';
        label.htmlFor = `sc-strategy-${i}`;
        label.textContent = `Firm ${i} (cost ${cost})`;
        const select = document.createElement('select');
        select.id = `sc-strategy-${i}`;
        select.className = 'config-input';
        STRATEGIES.forEach(([value, text]) => {
            const option = document.createElement('option');
            option.value = value;
            option.textContent = text;
            select.appendChild(option);
        });
        select.value = previous[i] || 'nash';
        container.append(label, select);
    });
}

function syncScenarioControls() {
    const demand = document.getElementById('sc-demand').value;
    const model = document.getElementById('sc-model').value;
    document.querySelectorAll('.sc-params').forEach(el => {
        el.classList.toggle('hidden', el.dataset.demand !== demand);
    });
    document.querySelectorAll('.config-check[data-model]').forEach(el => {
        el.classList.toggle('hidden', el.dataset.model !== model);
    });
    document.getElementById('sc-evolution-params').classList.toggle(
        'hidden', !document.getElementById('sc-evolution').checked
    );
}

function buildScenarioRequest() {
    const value = id => document.getElementById(id).value;
    const num = id => Number(value(id));
    const model = value('sc-model');
    const demand = value('sc-demand');
    const costs = parseCsv(value('sc-costs'));
    const body = {
        model,
        rounds: num('sc-rounds'),
        firms: costs.map(cost => ({ cost })),
        seed: value('sc-seed') === '' ? null : num('sc-seed')
    };

    if (demand === 'linear') {
        body.params = model === 'cournot'
            ? { a: num('sc-intercept'), b: num('sc-slope') }
            : { alpha: num('sc-intercept'), beta: num('sc-slope') };
        if (model === 'bertrand') {
            body.capacity_constraints = document.getElementById('sc-capacity').checked;
        }
    } else if (demand === 'isoelastic') {
        body.demand_type = 'isoelastic';
        body.params = { A: num('sc-iso-a'), elasticity: num('sc-iso-e') };
    } else {
        body.enhanced_demand = {
            demand_type: 'ces',
            elasticity: num('sc-ces-sigma'),
            market_elasticity: num('sc-ces-eta'),
            market_size: num('sc-ces-size')
        };
        const qualities = parseCsv(value('sc-ces-quality'));
        if (qualities.length) body.enhanced_demand.qualities = qualities;
    }

    const strategies = Array.from(document.querySelectorAll('#sc-strategies select'))
        .map((select, firm_id) => ({ firm_id, strategy_type: select.value }))
        .filter(s => s.strategy_type !== 'nash');
    if (strategies.length) body.advanced_strategies = strategies;

    if (document.getElementById('sc-evolution').checked) {
        body.market_evolution = {
            growth_rate: num('sc-growth'),
            entry_cost: num('sc-entry'),
            innovation_rate: num('sc-innovation')
        };
    }
    return body;
}

function formatError(detail) {
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
        return detail.map(d => `${(d.loc || []).slice(1).join('.')}: ${d.msg}`).join('; ');
    }
    return 'Simulation failed';
}

function formatNumber(v, digits = 2) {
    return v === null || v === undefined ? '—' : Number(v).toFixed(digits);
}

function seriesColor(i) {
    const isDark = document.body.getAttribute('data-theme') === 'dark';
    const color = SERIES_COLORS[i % SERIES_COLORS.length];
    return isDark && color === '#1D3557' ? '#58A6FF' : color;
}

function renderScenario() {
    if (!scenarioResult) return;
    const { run, events } = scenarioResult;
    const rounds = Object.keys(run.results).map(Number).sort((a, b) => a - b);
    const metrics = rounds.map(r => run.metrics[r] || run.metrics[String(r)]);
    const last = metrics[metrics.length - 1];

    document.getElementById('sc-price').textContent = formatNumber(last.market_price);
    document.getElementById('sc-hhi').textContent = formatNumber(last.hhi, 3);
    document.getElementById('sc-cs').textContent = formatNumber(last.consumer_surplus, 0);
    document.getElementById('sc-firms').textContent = `${metrics[0].num_firms} → ${last.num_firms}`;

    let datasets;
    if (scenarioSeries === 'actions' || scenarioSeries === 'profits') {
        const key = scenarioSeries === 'actions' ? 'action' : 'profit';
        const firms = new Set();
        rounds.forEach(r => Object.keys(run.results[r]).forEach(f => firms.add(f)));
        datasets = Array.from(firms).sort((a, b) => Number(a.split('_')[1]) - Number(b.split('_')[1]))
            .map((firm, i) => ({
                label: firm.replace('_', ' '),
                data: rounds.map(r => (run.results[r][firm] || {})[key] ?? null),
                borderColor: seriesColor(i),
                backgroundColor: seriesColor(i),
                borderWidth: 2,
                pointRadius: 0,
                spanGaps: false
            }));
    } else {
        datasets = [{
            label: scenarioSeries.replace('_', ' '),
            data: metrics.map(m => m[scenarioSeries]),
            borderColor: COLORS.primary,
            backgroundColor: COLORS.primary,
            borderWidth: 2,
            pointRadius: 0
        }];
    }
    charts.scenario.data.labels = rounds;
    charts.scenario.data.datasets = datasets;
    charts.scenario.update();

    const tbody = document.getElementById('sc-events');
    tbody.innerHTML = '';
    if (!events.length) {
        tbody.innerHTML = '<tr><td colspan="3" class="loading">No events in this run.</td></tr>';
        return;
    }
    events.forEach(e => {
        const row = document.createElement('tr');
        [e.round_idx, e.event_type.replace(/_/g, ' '), e.description].forEach(text => {
            const cell = document.createElement('td');
            cell.textContent = text;
            row.appendChild(cell);
        });
        tbody.appendChild(row);
    });
}

async function runScenario(event) {
    event.preventDefault();
    const error = document.getElementById('sc-error');
    const button = document.getElementById('sc-run');
    error.classList.add('hidden');
    button.disabled = true;
    button.textContent = 'Running…';
    try {
        const response = await fetch('/api/scenario', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(buildScenarioRequest())
        });
        const body = await response.json();
        if (!response.ok) throw new Error(formatError(body.detail));
        scenarioResult = body;
        renderScenario();
    } catch (err) {
        error.textContent = err.message;
        error.classList.remove('hidden');
    } finally {
        button.disabled = false;
        button.textContent = 'Run scenario';
    }
}

function initScenarioLab() {
    const ctx = document.getElementById('scenario-chart').getContext('2d');
    charts.scenario = new Chart(ctx, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: getChartConfig()
    });

    renderStrategyPickers();
    syncScenarioControls();
    document.getElementById('sc-costs').addEventListener('input', renderStrategyPickers);
    ['sc-demand', 'sc-model', 'sc-evolution'].forEach(id => {
        document.getElementById(id).addEventListener('change', syncScenarioControls);
    });
    document.getElementById('sc-demand').addEventListener('change', () => {
        // CES is differentiated price competition
        if (document.getElementById('sc-demand').value === 'ces') {
            document.getElementById('sc-model').value = 'bertrand';
            syncScenarioControls();
        }
    });
    document.getElementById('scenario-form').addEventListener('submit', runScenario);
    document.querySelectorAll('#sc-series .toggle-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('#sc-series .toggle-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            scenarioSeries = btn.dataset.series;
            renderScenario();
        });
    });
}

// One-click scenarios for first-time visitors
const PRESETS = {
    cartel: {
        model: 'bertrand', demand: 'ces', costs: '10, 10, 10', rounds: 60, seed: 7,
        strategies: ['cartel', 'collusive', 'nash'], evolution: false
    },
    learning: {
        model: 'cournot', demand: 'linear', costs: '10, 10', rounds: 80, seed: 3,
        strategies: ['fictitious_play', 'q_learning'], evolution: false
    },
    entry: {
        model: 'cournot', demand: 'isoelastic', costs: '10, 12', rounds: 80, seed: 9,
        strategies: ['nash', 'nash'], evolution: true, entry: 20, growth: 0.03, innovation: 0.3
    }
};

function applyPreset(name) {
    const p = PRESETS[name];
    const set = (id, v) => { document.getElementById(id).value = v; };
    set('sc-model', p.model);
    set('sc-demand', p.demand);
    set('sc-costs', p.costs);
    set('sc-rounds', p.rounds);
    set('sc-seed', p.seed);
    document.getElementById('sc-evolution').checked = p.evolution;
    if (p.evolution) {
        set('sc-entry', p.entry);
        set('sc-growth', p.growth);
        set('sc-innovation', p.innovation);
    }
    renderStrategyPickers();
    p.strategies.forEach((s, i) => set(`sc-strategy-${i}`, s));
    syncScenarioControls();
    document.getElementById('scenario-form').requestSubmit();
}

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initScenarioLab();
    document.querySelectorAll('#presets [data-preset]').forEach(btn => {
        btn.addEventListener('click', () => applyPreset(btn.dataset.preset));
    });
    applyPreset('cartel');
});
