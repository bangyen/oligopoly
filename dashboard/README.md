# Scenario Lab

A browser front end for the simulation engine. Pick a competition model and a
demand system (linear, isoelastic or CES). Give each firm a strategy: adaptive
Nash, a learning algorithm, or a cartel member. Optionally turn on market
evolution. The lab charts actions, profits, price, HHI and consumer surplus,
and lists cartel, entry, exit and innovation events.

```bash
pip install -e ".[api]"
python dashboard/main.py   # http://localhost:5050
```

Or with Docker: `just docker-dashboard`.

`POST /api/scenario` takes the same request body as the REST API's
`POST /simulate` and runs it against a private in-memory database, so the lab
supports exactly the options the API does.
