import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.utils import resample


def calculate_metrics(real_values, predicted_values, output_format, random_seed=0, bootstrap_iterations=100):
    np.random.seed(random_seed)

    stats = dict()

    stats["auc"] = lambda real, pred: metrics.roc_auc_score(real, pred)
    stats["sensitivity"] = lambda real, pred: metrics.recall_score(real, pred)
    stats["specificity"] = lambda real, pred: metrics.recall_score(real, pred, pos_label=0)
    stats["npv"] = lambda real, pred: metrics.precision_score(real, pred, pos_label=0)
    stats["ppv"] = lambda real, pred: metrics.precision_score(real, pred, pos_label=1)
    stats["accuracy"] = lambda real, pred: metrics.accuracy_score(real, pred)
    stats["balanced_accuracy"] = lambda real, pred: metrics.balanced_accuracy_score(real, pred)

    bootstrap_values = []

    for i in range(bootstrap_iterations):
        sample_real = resample(real_values, random_state=np.random.randint(0, bootstrap_iterations))
        sample_pred = resample(predicted_values, random_state=np.random.randint(0, bootstrap_iterations))

        current = dict()
        for k in stats.keys():
            current[k] = stats[k](sample_real, sample_pred)

        bootstrap_values.append(current)

    ci_l = pd.DataFrame.from_dict(bootstrap_values).quantile(0.025)
    ci_r = pd.DataFrame.from_dict(bootstrap_values).quantile(0.975)

    data = {
        'statistic': stats.keys(),
        'confidence interval': [(ci_l[x], ci_r[x]) for x in stats.keys()]
    }
    df = pd.DataFrame(data).set_index("statistic")

    if output_format == 'markdown':
        return df.to_markdown()

    return df

if __name__ == "__main__":
    real_values = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    predicted_values = [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0]
    print(calculate_metrics(real_values, predicted_values, "markdown"))
    print(calculate_metrics(real_values, predicted_values, "pandas"))