## Evalution dataset used for Tabular

The evaluation dataset is based on the open-source benchmark : https://bird-bench.github.io/
Click on Dev Set button: this will download the data locally.
The dataset used is Financial dataset.
All tables(7 tables in total) are used for evaluation except the "trans" table (due to its big size).
The names of the tables are: 'account.csv', 'card.csv', 'client.csv', 'disp.csv', 'district.csv', 'loan.csv', 'order.csv'
The tables are uploaded into S3 bucket in Redbox Dev environment. The name of the S3 bucket is 'redbox-evaluation-dataset' and prefix name is tabular.
