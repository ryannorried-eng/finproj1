# Debugging the Streamlit app (`line_tracker`)

## Reliable local setup (src-layout)

From the repository root:

```bash
python -m pip install -e .
```

Then run the app:

```bash
streamlit run src/line_tracker/dashboard.py
```

Using editable install is the preferred path because `streamlit run` executes a script file directly, and src-layout repos can otherwise resolve imports from an unrelated installed package.

## Quick diagnostics

### Confirm import origin

```bash
python -c "import line_tracker; print(line_tracker.__file__)"
```

You should see a path under this repo's `src/line_tracker`.

### Enable runtime import debugging in the app

```bash
LINE_TRACKER_DEBUG_IMPORTS=1 streamlit run src/line_tracker/dashboard.py --server.headless true
```

With this flag, the app prints the resolved `line_tracker.__file__` and shows a warning in the UI if the package origin doesn't match this checkout.

## Smoke tests

Run Streamlit smoke coverage via pytest:

```bash
pytest -q tests/test_streamlit_smoke.py
```

Or run the full suite:

```bash
pytest -q
```

## Known gotchas

- **src-layout + direct Streamlit script execution** can import from global site-packages instead of this checkout if not installed editable.
- To mitigate, prefer editable install and use the debug import flag above when troubleshooting path shadowing.
