const assert = require('node:assert/strict');
const fs = require('node:fs');
const {scoreTransaction, thresholdAtCapacity, evaluatePolicy} = require('../dist/scoring.js');
const data = JSON.parse(fs.readFileSync('dist/experiment.json'));
const vectors = JSON.parse(fs.readFileSync('reports/parity.json'));
let maxError = 0;
for (const name of Object.keys(data.models)) {
  vectors.features.forEach((values,i) => {
    const error = Math.abs(scoreTransaction(data.models[name],values)-vectors.expected[name][i]);
    maxError = Math.max(maxError,error);
    assert(error < 1e-9, `${name}: exported model differs by ${error}`);
  });
  const threshold = thresholdAtCapacity(data.validation_scores[name],1);
  assert(Math.abs(threshold-data.report.models[name].threshold)<1e-12);
  const policy = evaluatePolicy(data.test,name,threshold);
  for (const key of ['tp','fp','fn','tn','reviewed']) assert.equal(policy[key],data.report.models[name].test_policy[key]);
}
assert.throws(()=>scoreTransaction(data.models.logistic,[NaN]));
assert.throws(()=>thresholdAtCapacity([0.1],0));
const tiny = [{label:1,amount:100,scores:{m:0.9}},{label:0,amount:50,scores:{m:0.8}},
              {label:1,amount:20,scores:{m:0.1}},{label:0,amount:10,scores:{m:0}}];
const p=evaluatePolicy(tiny,'m',0.5,3,10,0.5);
assert.deepEqual([p.tp,p.fp,p.fn,p.tn,p.netSavings],[1,1,1,1,34]);
const splits=Object.values(data.report.split);
for(let i=1;i<splits.length;i++) assert(splits[i-1].time_end<splits[i].time_start);
console.log(`Model parity, temporal boundaries, threshold parity, confusion counts and cost arithmetic passed. Max scorer error: ${maxError}`);
