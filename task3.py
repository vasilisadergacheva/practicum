def get_best_threshold(targets, pred_probs):
    thresholds = sorted(list(set([0, 1] + pred_probs)))
    count = dict({threshold: [0, 0] for threshold in thresholds})

    positives = sum(targets)
    negatives = len(targets) - positives

    for target, prob in zip(targets, pred_probs):
        count[prob][target] += 1

    tpr = 1
    tnr = 0
    answer = (-1, -1) # (Youden's statistic, threshold (>))

    for threshold in thresholds:
        tpr -= count[threshold][1] / positives
        tnr += count[threshold][0] / negatives
        youden = tpr + tnr - 1

        if answer[0] < youden:
            answer = (youden, threshold)

    return answer


if __name__ == "__main__":
    print(get_best_threshold(
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0.5, 0.3, 0, 0.1, 0.5, 0.6, 0.9, 0.8]
    ))