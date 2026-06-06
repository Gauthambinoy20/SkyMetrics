# SkyMetrics — Roadmap

Expanding the British Airways Forage data-science project into a full,
tested airline-customer-intelligence product.

## Milestones

- [x] 0.1 Snapshot + package scaffold .................. 6%
- [x] 0.2 Baseline run (notebooks reproduce) .......... 12%
- [x] 0.3 Core logic extracted + unit tests ........... 20%
- [x] 0.4 SQLite persistence layer + tests ............ 28%
- [x] 0.5 Skytrax scraper (mocked tests) .............. 38%
- [x] 0.6 OpenSky + IAG external sources .............. 46%
- [x] 0.7 Transformer + aspect + topic NLP ............ 56%
- [x] 0.8 ML persistence + importance + model card .... 64%
- [x] 0.9 FastAPI service + tests ..................... 70%
- [x] 1.0 Streamlit dashboard + benchmarking .......... 80%
- [x] 1.1 Docker + compose (built, smoke-tested) ...... 84%
- [x] 1.2 CI/CD pipeline (all gates green locally) .... 90%
- [x] 1.3 Docs, diagrams, screenshots ................. 100%

## Phase 2 — AI & live serving

- [x] 2.0 Live BA flight tracking + map (OpenSky)
- [x] 2.1 Booking model served via /predict + transformer toggle
- [x] 2.2 LLM review summariser (OpenRouter free) + /summary
- [x] 2.3 RAG chatbot over reviews + /chat

## Test status

81 unit tests passing · network and LLM fully mocked (no API cost in CI).

## Known issues / limitations

- The transformer sentiment model downloads weights on first real use; it is
  injected/skipped in tests so CI never pulls a model.
- Notebook validation (`nbval`) is not wired into CI because the sentiment
  notebook scrapes the live Skytrax site; run it locally instead.
- Real private BA customer data is not publicly available — "customer data"
  here means public review/voice data plus the synthetic Forage booking set.

## Next

- [ ] Trustpilot + Reddit scrapers (same fetch/parse pattern as Skytrax)
- [ ] Time-series sentiment trend + volume forecasting
- [ ] Scheduled ETL populating the SQLite store
- [ ] WebSocket live flight stream to the map

## ✍️ TODO: my words

> Decision/opinion write-ups (why this architecture, what I learned, trade-offs)
> go here in my own words.
