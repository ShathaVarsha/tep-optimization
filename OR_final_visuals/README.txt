OR project visual outputs
=========================
Generated from the uploaded optimization_results_8vars.json and sensitivity_detailed_readable.csv.

Files:
- baseline_vs_optimized_summary.csv: main objective comparison
- process_response_comparison.csv: five predicted responses by scenario
- manipulated_variable_settings.csv: baseline and optimized manipulated-variable settings
- largest_local_sensitivity_by_response.csv: largest absolute local what-if change for each problem/response
- product_flow_comparison.png: product flow comparison chart
- operating_cost_comparison.png: approximate operating cost comparison chart
- process_response_percent_changes.png: percent changes in predicted responses
- local_sensitivity_comparison.png: largest local what-if response changes

Important limitations:
- These are surrogate-model predictions, not verified plant outcomes.
- The cost objective uses an approximate average purge price.
- Local what-if sensitivity is not causal analysis or formal LP shadow-price analysis.
- The JSON records pressure and compressor limits; those limits alone do not establish full process safety.
