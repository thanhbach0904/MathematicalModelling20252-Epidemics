# SARS Singapore reference data

These CSV files hold the reference data used by `spreader.sars_data`.

- `sars_singapore_secondary_cases.csv` stores the distribution used for the
  Fig. 14-style secondary-case comparison. The zero-secondary count and
  super-spreader descriptions are backed by CDC MMWR 52(18), "Severe Acute
  Respiratory Syndrome - Singapore, 2003". Intermediate bins are retained as
  digitised values from Fujie & Odagaki Fig. 14 because I did not find an
  official public table with the full distribution.
- `sars_singapore_epidemic_curve_6day.csv` stores the 6-day binned epidemic
  curve used for the Fig. 15 comparison. I found public source charts from CDC
  and Singapore surveillance reports, but not a machine-readable official
  onset-count table matching the paper's 6-day bins.

If a more authoritative table becomes available, replace the CSV contents while
keeping the same column names.
