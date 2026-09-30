// Functional fixture: not a substitute for real browser rendering or visual QA.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const {scoreTransaction,thresholdAtCapacity,evaluatePolicy}=require('../dist/scoring.js');
const html=fs.readFileSync('dist/index.html','utf8');
const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);
assert.equal(new Set(ids).size,ids.length,'HTML IDs must be unique');
const elements={};
function element(id){
  const el={id,value:'',textContent:'',hidden:false,dataset:{},options:[],listeners:{},
    classList:{toggle(){}},setAttribute(){},append(option){this.options.push(option);},
    addEventListener(name,fn){this.listeners[name]=fn;},checkValidity(){
      const v=Number(this.value);return Number.isFinite(v)&&
        (this.min===undefined||v>=this.min)&&(this.max===undefined||v<=this.max)&&
        (this.step!=='1'||Number.isInteger(v));
    }};
  Object.defineProperty(el,'value',{set(value){this._value=String(value);},get(){return this._value||'';}});
  Object.defineProperty(el,'innerHTML',{set(value){this.html=value;
    for(const m of value.matchAll(/<input id="(feature-\d+)"([^>]*)>/g)){
      const target=element(m[1]), attrs=m[2];target.value=(attrs.match(/value="([^"]+)"/)||[])[1]||'';
      for(const key of ['min','max']){const a=attrs.match(new RegExp(key+'="([^"]+)"'));if(a)target[key]=Number(a[1]);}
      target.step=(attrs.match(/step="([^"]+)"/)||[])[1];elements[m[1]]=target;
    }
  },get(){return this.html||'';}});return el;
}
for(const id of ids)elements[id]=element(id);
Object.assign(elements.capacity,{value:'1'});Object.assign(elements['review-cost'],{value:'3'});
Object.assign(elements['friction-cost'],{value:'10'});Object.assign(elements.recovery,{value:'80'});
const tabs=['policy','transaction','report'].map(id=>({...element(id+'-tab'),dataset:{tab:id}}));
const views=['policy','transaction','report'].map(id=>elements[id]);
const payload=JSON.parse(fs.readFileSync('dist/experiment.json'));
const registered=[];
const context=vm.createContext({console,Intl,Number,Math,Promise,Error,
  scoreTransaction,thresholdAtCapacity,evaluatePolicy,
  fetch:async()=>({ok:true,json:async()=>payload}),
  window:{addEventListener(){}},AbortController,
  document:{getElementById:id=>{assert(elements[id],`Missing element ${id}`);return elements[id];},
    querySelectorAll:selector=>selector==='.tab'?tabs:views,
    createElement:()=>element('option'),modelContext:{registerTool(tool){registered.push(tool);}}}});
vm.runInContext(fs.readFileSync('dist/app.js','utf8'),context);
(async()=>{
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(elements['load-error'].textContent,'');
  assert.equal(elements.recall.textContent,'15.3%');assert.equal(elements.precision.textContent,'23.4%');
  assert.equal(elements['savings'].textContent,'$146');
  const initialRecall=elements.recall.textContent;
  elements.capacity.value='5';elements.capacity.listeners.input();
  assert.notEqual(elements.recall.textContent,initialRecall);
  elements.example.value=payload.test[0].id;elements.example.listeners.change();
  assert.equal(Number(elements['feature-0'].value),payload.test[0].features[0]);
  elements['feature-0'].value='';elements['transaction-form'].listeners.submit({preventDefault(){}});
  assert.equal(elements['input-error'].textContent,'Enter valid values for every feature.');
  elements['feature-0'].value='100';elements['transaction-form'].listeners.submit({preventDefault(){}});
  assert.equal(elements['input-error'].textContent,'');
  tabs[2].listeners.click();assert.equal(elements.report.hidden,false);assert.equal(elements.policy.hidden,true);
  assert.equal(registered.length,1);
  const configured=registered[0].execute({model:'boosting',capacity_percent:2});
  assert.equal(configured.model,'boosting');assert.equal(elements.model.value,'boosting');
  assert.equal(elements['capacity-value'].textContent,'2.0%');
  assert.throws(()=>registered[0].execute({model:'invalid',capacity_percent:2}));
  assert.equal(elements.model.value,'boosting');
  console.log('UI fixture passed: initialization, policy controls, example inference, invalid inputs, tabs and structured tool logic. Actual browser/WebMCP context unavailable.');
})().catch(error=>{console.error(error);process.exitCode=1;});
