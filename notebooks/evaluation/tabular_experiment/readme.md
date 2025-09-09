## Steps to run the tabular evaluation
1. Run docker compose locally
2. Download the 7 tables (csv files) from the S3 bucket in Dev, named "redbox-evaluation-dataset" within Tabular prefix
3. Upload locally these 7 tables into Redbox using the UI 
4. Run the evaluation script: tabular_evaluation.py. Once the script finishes, the final results are stored in notebooks/evaluation/data_results/tabular/evaluation_results.json
5. Run the performance notebook (calculate_performance_metrics.ipynb) to calculate performance metrics based on the evaluation results.
6. The data output from each SQL query (based on ground truth as well as tabular agent) are saved as text files in notebooks/evaluation/data_results/tabular/. These files are not commited to Git but are available for local inspection.

For information about the evaluation dataset, please refer to: notebooks/evaluation/data_results/tabular/bird-dataset.md