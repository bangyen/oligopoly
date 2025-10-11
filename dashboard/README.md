# Oligopoly Analytics Dashboard

A minimalist, Bauhaus-inspired dashboard for visualizing oligopoly market simulations.

## Running the Dashboard

```bash
# From project root
source venv/bin/activate
python3 dashboard/main.py
```

Visit **http://localhost:5050** in your browser.

## Features

### Views

- **Overview**: Market metrics, profit distribution, and market share visualization
- **Cournot**: Quantity competition dynamics over 50 rounds
- **Bertrand**: Price competition dynamics over 50 rounds  
- **Metrics**: Detailed Nash equilibrium calculations and firm data

### Interactions

- Navigation sidebar for switching between views
- Toggle buttons on charts to switch between quantities/prices/profits
- Refresh button to reload all data
- Run Simulation button to execute new simulations

## Design System

**Typography**: Space Grotesk  
**Colors**: Red (#E63946), Navy (#1D3557), Steel Blue (#457B9D)  
**Style**: Flat colors, sharp edges, custom SVG icons, geometric layouts

## API Endpoints

- `GET /` - Dashboard UI
- `GET /api/metrics` - Nash equilibrium metrics
- `GET /api/simulation/cournot` - 50-round Cournot simulation
- `GET /api/simulation/bertrand` - 50-round Bertrand simulation

## Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Charts**: Chart.js
- **Design**: Custom CSS with design tokens

