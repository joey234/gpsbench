# Project website

Open `index.html` through a static web server. The site reads generated `data.js` and uses Chart.js for plots.

- Edit task/model descriptions in `config.json`.
- Current model scores come from `../results/current/scores.json`.
- Regenerate `data.js` with `scripts/export_leaderboard.py` from the repository root.

The site reports 17 main tasks, 58,614 examples, and 17,342 test examples per model. Missing Data is an additional probe and does not enter these averages. Numerical task and overall scores are labelled as scores rather than accuracy percentages.
