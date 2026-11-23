# Premier League Predictor

Predicts the next Premier League fixtures (win/draw/loss chances, a predicted score, close
alternatives and a safer "win or draw" call) and shows the evidence behind each one: form,
xG, team news, head-to-head and how the model's numbers were built up. It also keeps score
of its own predictions.

Not betting advice. Football is noisy and single-match predictions are often wrong.

## Refresh

Double-click `refresh.bat`, or run:

```
%LOCALAPPDATA%\Programs\Python\Python312\python.exe build.py
```

This downloads fresh data, refits the model, saves predictions for games that haven't kicked
off (`predictions_log.json`), grades finished ones, and writes:

- `out/index.html`: the full page with data built in. Open it straight in a browser.
- `out/artifact.html` + `out/data.json`: the same page for the claude.ai artifact. Ask Claude
  to republish it after a refresh.

`python build.py --offline` rebuilds from already-downloaded data. `python tune.py` reruns
the walk-forward test used to pick the model settings.

## Files

| File | What it does |
|---|---|
| `fetch.py` | Downloads the data, records download times in `data/raw/meta.json` |
| `data.py` | Reads and merges results, fixtures, odds and team news |
| `model.py` | Dixon-Coles goals model (weighted, with xG blend and priors) |
| `build.py` | Predictions, team news adjustment, form, head-to-head, backtest, log, page |
| `tune.py` | Walk-forward test of model settings |
| `squad.py` | Player attack ratings, form and line-up strength |
| `tune_squad.py` | Walk-forward test of squad strength and line-ups |
| `local_copies.json` | Folders that get a fresh copy of the page after every build (e.g. a Desktop folder) |
| `jobs.py` | Cloud job helper: page buttons and match-day notifications |
| `.github/workflows/predictor.yml` | The GitHub Actions schedule |
| `notify.py` | Phone notifications through ntfy (`python notify.py --test`); topic in `notify_config.json` (keep private) |
| `tune_confidence.py` | Hit rates by confidence; sets the "strongest picks" cut-offs |
| `page/template.html` | The page; `build.py` fills in the data |

## Data sources

- **football-data.co.uk**: results, xG, shots and bookmaker odds as free CSV files
  (2019-20 onwards, for ratings and head-to-head). The CSV updates a couple of times a week.
- **Fantasy Premier League public data** (`fantasy.premierleague.com/api/...`): fixture list,
  kickoff times, the newest scores, and player injury/suspension news with chance of playing.
  Public JSON, no login or key; read twice per refresh for personal use.

No API keys or accounts are used.

## The model in short

1. Attack and defence ratings for every team from the last two seasons, with recent games
   weighted more (half-life 180 days) and this season's games using 70% xG / 30% goals.
2. Home advantage fitted from the data; promoted sides start from a slightly weaker prior.
3. Team news: each missing player removes half of his share of team xG+xA (attack) or a
   share of back-line minutes (defence), scaled by his chance of playing.
4. Dixon-Coles score matrix gives every scoreline's chance, and from that win/draw/loss.
5. `markets.py`: goal lines, both teams to score, clean sheets, margins, first goal and
   half-time (from the same expected goals); corners and cards from their own team ratings
   (cards also adjust for the referee, when the free odds file names him), spread with a
   negative binomial. Shots on target work the same way. Each headline market is
   backtested against a simple base-rate guess.
6. `scorers.py`: goalscorer chances. Each player's xG-based rate per 90 × expected minutes
   (recent games × chance of playing) shares out the team's expected goals. Uses the FPL
   per-gameweek feed (`event/N/live`), cached once a gameweek is finished, and is backtested
   with a calibration table.

7. When bookmaker odds are out, the headline result chances are 40% model + 60% bookmakers
   (only the home/away balance is taken from the odds; total goals stay the model's).
   `python tune_data.py` tests this and the other extra data tried (shot-based xG for
   older seasons, Championship games for promoted sides). Those two didn't help, so they
   aren't used in the ratings; Championship meetings are shown in head-to-head.

8. `squad.py`: player ratings and squad strength from the FPL per-match stats (minutes,
   starts, xG, xA) and FPL starting prices. Each outfield player gets an attack rating
   (45-99, percentile of xG + 0.7 xA per 90 within his position); "hot"/"cold" form compares
   his last 4 games with his season. FIFA/EA ratings aren't used (EA's data, not licensed).
   `python tune_squad.py` tests squad strength on this season: the squad seen days before
   kickoff didn't help reliably, but confirmed line-ups did (model RPS 0.2067 -> 0.2043, one
   more right result). So each match on the page has a line-up check: tick the starters and
   the result chances update in the browser.

9. Strongest picks: the calls that come in most often once the model is sure enough.
   Result at 55%+, favourite-or-draw at 80%+, more shots on target at 65%+ and more corners
   at 65%+. `python tune_confidence.py` picked these on 2025-26 (350 games, walk-forward:
   62% / 89% / 77% / 67% came in) and checked them on this season (65% / 93% / 79% / 70%).
   It writes `confidence_results.json`, which the build reads for last season's numbers.
   Strong picks are saved in `predictions_log.json` and graded like the main call.

10. Odds freshness: football-data.co.uk's `fixtures.csv` is updated a couple of times a
    week and not again before kickoff. `fetch.py` records its Last-Modified time, and each
    match shows how old its odds are. Over 1,900 games (2021-22 to 2025-26), closing odds
    picked 55.9% of results against 55.3% for the earlier ones. Closing odds aren't free
    before kickoff, so refresh soon after the file updates (so each match has odds), then
    again on match day for team news.

11. Fair odds and the odds check: each strong pick shows its fair odds (1 / our chance;
    for corners and shots on target also the fair odds if a tie loses). On the Fixtures tab,
    "Odds check" takes the odds you see at a bookmaker and a stake of 20-100 kr, and shows
    the profit if it wins, the chance of losing the stake and the average result per bet. It
    also compares a combined bet (picks from different matches) with the same stake as
    singles. Stake and typed odds are kept in your browser only. Nothing is bet automatically.

12. Bet slip, limits and breaks (published page only). After placing a strong pick on Oddset
    yourself, tap "Yes, I placed it" (or "No, skip"). Answers and limits are saved in the
    artifact's database under your own private path (`data/users/<you>/limits` and its
    `bets` collection), which only you can read. Limits: stakes 20-100 kr, a weekly budget
    (default 200 kr) and a monthly loss limit (default 300 kr, counting open bets). A break
    (24 hours to 3 months) hides the betting parts and stops bet notifications. Lowering a
    limit or starting a break works at once; raising a limit or ending a break early waits
    72 hours. Nothing is reported anywhere. Nothing logs in to Danske Spil: their terms
    forbid robots that automate the game, so you check the odds and place bets yourself.

Backtest of the model alone this season (refit before each gameweek, no peeking): right result 50/107 (47%),
bookmakers 51/107; RPS 0.207 vs bookmakers 0.206 (lower is better). Team news isn't in the
backtest because past injury lists aren't archived.

## Hosted version (no Claude needed)

The page is published at https://mixeholton.github.io/pl-predictor/ by GitHub Actions
(`.github/workflows/predictor.yml`), which runs on GitHub's servers, so your PC can be off:

- every morning at 08:00 UTC: refresh, then notify today's strong picks on match days
  (`python jobs.py matchday`, which respects your break and limits);
- Tuesdays and Fridays at 12:10 UTC: refresh after the new bookmaker odds;
- every 15 minutes 07:00-21:45 UTC: answer the page's "Refresh predictions" and "Send test
  notification" buttons (`python jobs.py check`).

Your bet log, limits and break are saved in a free Supabase project (table `docs`, one row per
item, Row Level Security so only you can read yours). Sign in on the My bets tab. The cloud job
reads them through the token-protected database function `feed`, never with a service key.

GitHub secrets the workflow needs: `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `FEED_TOKEN` (matches
`private.settings.feed_token` in Supabase) and `NTFY_TOPIC` (your phone-notification topic).
`site_config.json` holds the public page URL and Supabase anon key (safe to publish).
Oddset prices still come only from what you type, or from asking Claude to check by hand:
nothing reads Oddset automatically (Danske Spil's terms forbid robots).
