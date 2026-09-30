/* Exported sklearn estimators; identical preprocessing and sigmoid calibration. */
(function(root) {
  function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
  function scoreTransaction(model, values) {
    if (values.length !== model.features.length || values.some(v => !Number.isFinite(v))) {
      throw new Error('Every feature must be a finite number.');
    }
    let margin = model.intercept;
    if (model.kind === 'logistic') {
      values.forEach((v, i) => { margin += model.coefficients[i] * (v - model.mean[i]) / model.scale[i]; });
    } else {
      for (const tree of model.trees) {
        let node = 0;
        while (tree.left[node] !== -1) {
          // sklearn decision trees compare float32 input to float64 thresholds.
          node = Math.fround(values[tree.feature[node]]) <= tree.threshold[node] ? tree.left[node] : tree.right[node];
        }
        margin += model.learning_rate * tree.value[node];
      }
    }
    const p = Math.max(1e-6, Math.min(1 - 1e-6, sigmoid(margin)));
    return sigmoid(model.calibration.intercept + model.calibration.coefficient * Math.log(p / (1 - p)));
  }
  function thresholdAtCapacity(scores, percent) {
    if (!Number.isFinite(percent) || percent <= 0 || percent >= 100) throw new Error('Invalid review target.');
    const sorted = scores.slice().sort((a, b) => a - b);
    return sorted[Math.ceil((sorted.length - 1) * (1 - percent / 100))];
  }
  function evaluatePolicy(rows, name, threshold, reviewCost=3, frictionCost=10, recovery=1) {
    let tp=0, fp=0, fn=0, tn=0, caughtAmount=0, missedAmount=0;
    for (const row of rows) {
      const flagged = row.scores[name] >= threshold;
      if (row.label) { if (flagged) { tp++; caughtAmount += row.amount; } else { fn++; missedAmount += row.amount; } }
      else if (flagged) fp++; else tn++;
    }
    const reviewed = tp + fp;
    return {tp,fp,fn,tn,reviewed,precision:tp/(reviewed||1),recall:tp/(tp+fn||1),
      reviewRate:reviewed/rows.length,caughtAmount,missedAmount,
      netSavings:caughtAmount*recovery-reviewed*reviewCost-fp*frictionCost};
  }
  const api = {scoreTransaction, thresholdAtCapacity, evaluatePolicy};
  if (typeof module !== 'undefined') module.exports = api;
  Object.assign(root, api);
})(typeof window !== 'undefined' ? window : globalThis);
