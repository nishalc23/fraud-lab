'use strict';
let experiment, currentThreshold, currentPolicy, currentModel='logistic';
const $=id=>document.getElementById(id);
const names={logistic:'Logistic regression',boosting:'Gradient boosting'};
const pct=(v,d=1)=>(v*100).toFixed(d)+'%';
const num=v=>v.toLocaleString('en-US');
const money=v=>new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:0}).format(v);
const label=name=>name.replaceAll('_',' ').replace(/\b\w/g,c=>c.toUpperCase());
let ranked={}, chartPolicies={}, sortedValidation={};
function tabView(id){
  document.querySelectorAll('.view').forEach(el=>el.hidden=el.id!==id);
  document.querySelectorAll('.tab').forEach(el=>{const active=el.dataset.tab===id;el.classList.toggle('active',active);el.setAttribute('aria-pressed',String(active));});
}
function cutoff(name,percent){const a=sortedValidation[name];return a[Math.ceil((a.length-1)*(1-percent/100))];}
function getCosts(){return {review:Number($('review-cost').value),friction:Number($('friction-cost').value),recovery:Number($('recovery').value)/100};}
function updatePolicy(){
  currentModel=$('model').value;
  const capacity=Number($('capacity').value), costs=getCosts();
  if(!Number.isFinite(costs.review)||!Number.isFinite(costs.friction)||costs.review<0||costs.friction<0)return;
  currentThreshold=cutoff(currentModel,capacity);
  currentPolicy=evaluatePolicy(experiment.test,currentModel,currentThreshold,costs.review,costs.friction,costs.recovery);
  const p=currentPolicy, prevalence=experiment.test.filter(r=>r.label).length/experiment.test.length;
  $('capacity-value').textContent=capacity.toFixed(1)+'%';$('recovery-value').textContent=pct(costs.recovery,0);
  $('recall').textContent=pct(p.recall);$('precision').textContent=pct(p.precision);$('review-rate').textContent=pct(p.reviewRate,2);$('savings').textContent=money(p.netSavings);
  $('caught-count').textContent=`${p.tp} of ${p.tp+p.fn} fraud cases`;
  $('lift').textContent=`${(p.precision/prevalence).toFixed(1)}× test fraud prevalence`;
  $('review-count').textContent=`${num(p.reviewed)} transactions flagged`;
  for(const key of ['tp','fp','fn','tn'])$(key).textContent=num(p[key]);
  $('prevented').textContent=money(p.caughtAmount*costs.recovery);
  $('review-expense').textContent=money(p.reviewed*costs.review);$('friction-expense').textContent=money(p.fp*costs.friction);
  $('selection-note').textContent=experiment.report.selected_model===currentModel?'Selected on validation at the original 1% target.':'Comparison model; original validation selection favored '+names[experiment.report.selected_model]+'.';
  $('queue').innerHTML=ranked[currentModel].slice(0,8).map(r=>`<tr><td><button class="queue-link" data-id="${r.id}" style="border:0;background:none;padding:0;color:#087f6c;font:inherit;text-decoration:underline;cursor:pointer">${r.id}</button></td><td>${money(r.amount)}</td><td>${pct(r.scores[currentModel],2)}</td><td><span class="pill ${r.scores[currentModel]>=currentThreshold?'review':''}">${r.scores[currentModel]>=currentThreshold?'Review':'Pass'}</span></td><td><span class="pill ${r.label?'fraud':''}">${r.label?'Fraud':'Legitimate'}</span></td></tr>`).join('');
  drawChart(capacity);scoreForm();
}
function drawChart(capacity){
  const svg=$('coverage-chart'), W=700,H=270,L=52,R=22,T=20,B=42;
  const x=v=>L+v/10*(W-L-R),y=v=>H-B-v*(H-T-B);
  let html='';
  for(let i=0;i<=4;i++){const value=i/4;html+=`<line x1="${L}" y1="${y(value)}" x2="${W-R}" y2="${y(value)}" stroke="#e3e9f1"/><text x="${L-12}" y="${y(value)+4}" text-anchor="end" fill="#56657a" font-size="12">${pct(value,0)}</text>`;}
  for(let i=0;i<=10;i+=2)html+=`<text x="${x(i)}" y="${H-B+24}" text-anchor="middle" fill="#56657a" font-size="12">${i}%</text>`;
  html+=`<text x="${(W+L-R)/2}" y="${H-2}" text-anchor="middle" fill="#56657a" font-size="12">Validation review target</text>`;
  for(const name of ['logistic','boosting']){
    const color=name==='logistic'?'#087f6c':'#4672e5';
    const points=chartPolicies[name].map(p=>`${x(p.capacity)},${y(p.recall)}`).join(' ');
    html+=`<polyline points="${points}" fill="none" stroke="${color}" stroke-width="2.5"/>`;
  }
  html+=`<line x1="${x(capacity)}" y1="${T}" x2="${x(capacity)}" y2="${H-B}" stroke="#b57d0a" stroke-dasharray="4 4"/><circle cx="${x(capacity)}" cy="${y(currentPolicy.recall)}" r="6" fill="${currentModel==='logistic'?'#087f6c':'#4672e5'}" stroke="white" stroke-width="2"/>`;
  svg.innerHTML=html;
}
function setExample(value){
  const row=experiment.test.find(r=>r.id===value), values=row?row.features:experiment.models[currentModel].median;
  values.forEach((v,i)=>$('feature-'+i).value=v);scoreForm();
}
function scoreForm(){
  if(!experiment||!currentThreshold)return;
  const model=experiment.models[currentModel],values=model.features.map((_,i)=>Number($('feature-'+i).value));
  const invalid=model.features.some((name,i)=>{
    const el=$('feature-'+i),v=values[i];
    return el.value.trim()===''||!Number.isFinite(v)||!el.checkValidity();
  });
  if(invalid){$('input-error').textContent='Enter valid values for every feature.';return;}
  $('input-error').textContent='';
  const score=scoreTransaction(model,values),review=score>=currentThreshold;
  $('explorer-model').textContent=names[currentModel];$('risk-score').textContent=pct(score,2);
  $('risk-action').textContent=review?'Flag for review':'Pass under current policy';$('risk-action').classList.toggle('review',review);
  $('risk-threshold').textContent=`Current cutoff: ${pct(currentThreshold,2)} · ${$('capacity').value}% validation target`;
  const sensitivity=model.features.map((name,i)=>{const alternative=values.slice();alternative[i]=model.median[i];return {name,change:score-scoreTransaction(model,alternative)};}).sort((a,b)=>Math.abs(b.change)-Math.abs(a.change));
  $('sensitivity').innerHTML=sensitivity.slice(0,5).map(r=>`<div class="sensitivity-row"><span>${label(r.name)}</span><strong>${r.change>=0?'+':''}${(r.change*100).toFixed(2)} pp</strong></div>`).join('');
}
function renderReport(){
  const r=experiment.report;
  $('selection-rule').textContent=r.selection_rule+' Selected: '+names[r.selected_model]+'.';
  $('split-track').innerHTML=Object.entries(r.split).map(([name,p])=>`<div style="flex:${p.rows}">${label(name)}<br>${Math.round(p.rows/r.rows*100)}%</div>`).join('');
  $('split-details').innerHTML='<table><thead><tr><th>Partition</th><th>Rows</th><th>Fraud</th><th>Time range (hours)</th></tr></thead><tbody>'+Object.entries(r.split).map(([name,p])=>`<tr><td>${label(name)}</td><td>${num(p.rows)}</td><td>${num(p.fraud)}</td><td>${(p.time_start/3600).toFixed(1)}–${(p.time_end/3600).toFixed(1)}</td></tr>`).join('')+'</tbody></table>';
  $('model-table').innerHTML='<table><thead><tr><th>Metric</th><th>Logistic</th><th>Boosting</th></tr></thead><tbody>'+[
    ['Validation AP',...['logistic','boosting'].map(n=>r.models[n].validation_ap.toFixed(3))],
    ['Test AP',...['logistic','boosting'].map(n=>r.models[n].test_ap.toFixed(3))],
    ['Test ROC AUC',...['logistic','boosting'].map(n=>r.models[n].test_roc_auc.toFixed(3))],
    ['Test Brier score',...['logistic','boosting'].map(n=>r.models[n].test_brier.toFixed(4))],
    ['Recall · original policy',...['logistic','boosting'].map(n=>pct(r.models[n].test_policy.recall))],
    ['Recall · bootstrap 95% interval',...['logistic','boosting'].map(n=>r.models[n].test_recall_interval.map(v=>pct(v)).join('–'))],
    ['Precision · original policy',...['logistic','boosting'].map(n=>pct(r.models[n].test_policy.precision))],
    ['Test review rate',...['logistic','boosting'].map(n=>pct(r.models[n].test_policy.review_rate,2))],
  ].map(row=>'<tr>'+row.map((v,i)=>`<${i?'td':'th'}>${v}</${i?'td':'th'}>`).join('')+'</tr>').join('')+'</tbody></table>';
  $('audit').innerHTML=`<p>${num(r.rows)} transactions · ${num(r.fraud_count)} fraud cases (${pct(r.fraud_rate,2)}). ${num(r.duplicates_removed)} exact duplicates removed before partitioning. No missing model inputs. Random seed: ${r.seed}.</p><p>Training preprocessing is fit on the training partition only. Class labels and elapsed time are excluded from model inputs. Calibration, validation, and test keep their natural class prevalence.</p><p class="control-help">Python ${r.environment.python} · scikit-learn ${r.environment.scikit_learn}<br>Dataset fingerprint: ${r.data_sha256.slice(0,16)}…</p>`;
  $('provenance').textContent=r.source==='synthetic'?'60,000 simulated events generated by a seeded Python process. Risk labels include hidden variation and random outcomes; device frequency and risk drift with time. These are not real customers or Ramp data, and performance cannot establish real-world fraud detection quality.':'Original anonymized ULB/Worldline credit card benchmark, retrieved through the TensorFlow mirror. V1–V28 are upstream PCA features with unknown fitting provenance; their real-world meanings are unavailable. This short historical sample cannot establish performance on Ramp traffic.';
}
async function initialize(){
  if(window.EMBEDDED_EXPERIMENT) experiment=window.EMBEDDED_EXPERIMENT;
  else {const response=await fetch('experiment.json');if(!response.ok)throw Error('Experiment data could not be loaded.');experiment=await response.json();}
  const r=experiment.report;
  for(const name of Object.keys(experiment.models)){
    sortedValidation[name]=experiment.validation_scores[name].slice().sort((a,b)=>a-b);
    ranked[name]=experiment.test.slice().sort((a,b)=>b.scores[name]-a.scores[name]);
    chartPolicies[name]=[];
    for(let i=1;i<=100;i++){const c=i/10;chartPolicies[name].push({capacity:c,recall:evaluatePolicy(experiment.test,name,cutoff(name,c)).recall});}
  }
  $('dataset-badge').textContent=r.source==='synthetic'?'SIMULATED DATA':'ULB / WORLDLINE BENCHMARK';
  $('source-note').textContent=r.source==='synthetic'?'Simulation benchmark. All metrics below come from actually trained models on simulated transactions. This demonstrates the workflow; it does not measure real-world fraud detection.':'Historical benchmark. Metrics come from anonymized ULB/Worldline transactions, not Ramp customer data. Results are offline and do not establish production performance.';
  $('snapshot-count').textContent=num(experiment.test.length)+' transactions';$('snapshot-fraud').textContent=num(r.split.test.fraud)+' observed fraud cases';$('footer-source').textContent=r.source==='synthetic'?'Simulated data · seed 42':'Historical public benchmark';
  currentModel=r.selected_model;$('model').value=currentModel;
  const model=experiment.models[currentModel];
  $('feature-inputs').innerHTML=model.features.map((name,i)=>{
    const bool=['new_device','new_merchant','international'].includes(name),integer=['account_age_days','transactions_1h','failed_attempts_24h','hour'].includes(name);
    const limits=r.source==='synthetic'?`min="0" ${bool?'max="1"':name==='hour'?'max="23"':''}`:'';
    return `<div><label for="feature-${i}">${label(name)}</label><input id="feature-${i}" type="number" required ${limits} step="${bool||integer?'1':'any'}" value="${model.median[i]}"></div>`;
  }).join('');
  const examples=[...ranked[currentModel].slice(0,5),...experiment.test.filter(row=>!row.label).slice(0,3)];
  for(const row of examples){const option=document.createElement('option');option.value=row.id;option.textContent=`${row.id} · ${money(row.amount)} · observed ${row.label?'fraud':'legitimate'}`;$('example').append(option);}
  renderReport();updatePolicy();
  document.querySelectorAll('.tab').forEach(btn=>btn.addEventListener('click',()=>tabView(btn.dataset.tab)));
  for(const id of ['model','capacity','review-cost','friction-cost','recovery'])$(id).addEventListener('input',updatePolicy);
  $('example').addEventListener('change',()=>setExample($('example').value));
  $('transaction-form').addEventListener('submit',e=>{e.preventDefault();scoreForm();});
  $('queue').addEventListener('click',event=>{const btn=event.target.closest('[data-id]');if(!btn)return;const id=btn.dataset.id;if(!Array.from($('example').options).some(o=>o.value===id)){const option=document.createElement('option');option.value=id;option.textContent=id;$('example').append(option);}$('example').value=id;setExample(id);tabView('transaction');});
  registerTools();
}
function registerTools(){
  const context=document.modelContext;if(!context?.registerTool)return;
  const lifecycle=new AbortController();window.addEventListener('pagehide',()=>lifecycle.abort(),{once:true});
  Promise.resolve(context.registerTool({name:'configure_review_policy',title:'Configure fraud review policy',description:'Change the visible model and validation review target; return held-out test policy metrics. This is an offline simulation.',inputSchema:{type:'object',properties:{model:{type:'string',enum:['logistic','boosting']},capacity_percent:{type:'number',minimum:0.1,maximum:10}},required:['model','capacity_percent'],additionalProperties:false},annotations:{readOnlyHint:false,untrustedContentHint:false},execute(input){if(!input||!['logistic','boosting'].includes(input.model)||!Number.isFinite(input.capacity_percent)||input.capacity_percent<0.1||input.capacity_percent>10||Object.keys(input).some(k=>!['model','capacity_percent'].includes(k)))throw Error('Invalid policy configuration');$('model').value=input.model;$('capacity').value=String(Math.round(input.capacity_percent*10)/10);tabView('policy');updatePolicy();return {model:currentModel,validation_target_percent:Number($('capacity').value),threshold:currentThreshold,...currentPolicy};}},{signal:lifecycle.signal})).catch(()=>{});
}
initialize().catch(error=>{$('load-error').hidden=false;$('load-error').textContent=error.message+' Serve this project with python3 -m http.server --directory dist.';$('source-note').textContent='Experiment unavailable.';});
