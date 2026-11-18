# Iris dataset

`iris.csv` is the local, offline copy used by both the CLI and dashboard. It contains the canonical 150 records from scikit-learn's bundled `load_iris` dataset, derived from the classic Fisher Iris dataset. It is not simulated or generated example data. The CSV export was checked against the bundled scikit-learn records feature-by-feature and label-by-label.

## Schema

| Column | Meaning | Unit/type |
| --- | --- | --- |
| `sepal length (cm)` | Sepal length | centimeters |
| `sepal width (cm)` | Sepal width | centimeters |
| `petal length (cm)` | Petal length | centimeters |
| `petal width (cm)` | Petal width | centimeters |
| `species` | Recorded Iris class | `setosa`, `versicolor`, `virginica` |

Each class has 50 records. The dataset has no missing values. There is one exact duplicate complete row in the source dataset; it is preserved as part of the canonical records, not deduplicated. Automated tests compare both the records and this duplicate count with scikit-learn's bundled reference.

This small educational dataset is not evidence that a classifier is suitable for a production, medical, or other high-stakes use case.
