const state={metadata:null,result:null};
const $=(s)=>document.querySelector(s);
const esc=(v)=>String(v).replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));

async function init(){
  try{
    const res=await fetch('/api/metadata',{cache:'no-store'}); state.metadata=await res.json();
    renderAcute(); renderFields(); renderEvidence(); bind();
  }catch(e){ $('#modelStamp').innerHTML='<span>MODEL STATUS</span><strong>연결 오류</strong><small>서버를 다시 시작해 주세요</small>'; }
}
function renderAcute(){
  $('#acuteFields').innerHTML=state.metadata.acute.map(x=>`<label><input type="checkbox" name="acute.${esc(x.name)}"><span>${esc(x.label)}</span></label>`).join('');
}
function renderFields(){
  $('#fieldGrid').innerHTML=state.metadata.fields.map(f=>{
    const optional=f.required?'':'<span class="optional-tag">선택</span>';
    const label=`<label for="${f.name}">${esc(f.label)}${optional}${f.required?'<small>필수</small>':''}</label>`;
    let control='';
    if(f.type==='select'){
      control=`<select id="${f.name}" name="${f.name}" ${f.required?'required':''}><option value="">선택하세요</option>${f.options.map(o=>`<option value="${esc(o.value)}">${esc(o.label)}</option>`).join('')}</select>`;
    }else{
      control=`<input id="${f.name}" name="${f.name}" type="number" min="${f.min}" max="${f.max}" step="${f.step}" ${f.required?'required':''} inputmode="decimal"><span class="unit">${esc(f.unit||'')}</span>`;
    }
    return `<div class="field"><div>${label}</div><div class="control-wrap">${control}</div>${f.help?`<div class="field-help">${esc(f.help)}</div>`:''}</div>`;
  }).join('');
}
function renderEvidence(){
  const m=state.metadata.model;
  $('#modelStamp').innerHTML=`<span>MODEL STATUS</span><strong>준비됨</strong><small>${esc(m.version)}<br>${m.training_rows.toLocaleString()}건 학습</small>`;
  $('#rowsMetric').textContent=m.training_rows.toLocaleString();
  $('#aucMetric').textContent=m.auroc_oof.toFixed(3);
  $('#sensitivityMetric').textContent=(m.sensitivity*100).toFixed(1)+'%';
}
function bind(){
  $('#riskForm').addEventListener('submit',submitForm);
  $('#loadSample').addEventListener('click',loadSample);
  $('#resetResult').addEventListener('click',resetResult);
  $('#printReport').addEventListener('click',()=>window.print());
  $('#downloadCsv').addEventListener('click',downloadCsv);
  document.querySelectorAll('[name^="acute."]').forEach(x=>x.addEventListener('change',()=>{if(x.checked) x.closest('label').style.borderColor='#a33a35'; else x.closest('label').style.borderColor='';}));
}
function loadSample(){
  const sample={age:58,sex:1,cp:4,trestbps:142,chol:240,fbs:0,restecg:0,thalach:132,exang:1,oldpeak:1.8,slope:2,ca:1,thal:7};
  Object.entries(sample).forEach(([k,v])=>{const el=document.getElementById(k);if(el)el.value=v;});
  $('#acknowledge').checked=true; $('#formErrors').hidden=true;
}
function collect(){
  const payload={acute:{}};
  state.metadata.fields.forEach(f=>{const el=document.getElementById(f.name); payload[f.name]=el.value===''?null:Number(el.value);});
  document.querySelectorAll('[name^="acute."]').forEach(el=>payload.acute[el.name.split('.')[1]]=el.checked);
  return payload;
}
async function submitForm(e){
  e.preventDefault(); const errors=[];
  if(!$('#acknowledge').checked) errors.push('연구용 결과의 사용범위를 확인해 주세요.');
  state.metadata.fields.filter(f=>f.required).forEach(f=>{if(document.getElementById(f.name).value==='') errors.push(`${f.label} 항목을 입력해 주세요.`);});
  if(errors.length){showErrors(errors);return;}
  const btn=$('.primary-button');btn.disabled=true;btn.querySelector('span').textContent='분류하는 중…';
  try{
    const res=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(collect())});
    const data=await res.json(); if(!data.ok){showErrors(data.errors||['입력값을 확인해 주세요.']);return;} state.result=data;renderResult(data);
  }catch(err){showErrors(['서버와 연결할 수 없습니다. 프로그램을 다시 시작해 주세요.']);}
  finally{btn.disabled=false;btn.querySelector('span').textContent='위험도 분류하기';}
}
function showErrors(errors){const box=$('#formErrors');box.innerHTML='<strong>입력을 확인해 주세요.</strong><br>'+errors.map(esc).join('<br>');box.hidden=false;box.scrollIntoView({behavior:'smooth',block:'center'});}
function renderResult(r){
  $('#formErrors').hidden=true;$('#resultEmpty').hidden=true;$('#resultContent').hidden=false;
  $('#riskPercent').textContent=r.percentage.toFixed(1)+'%';$('#riskTier').textContent=r.tier_ko;
  const low=(r.thresholds.low_attention*100).toFixed(1),high=(r.thresholds.attention_high*100).toFixed(1);
  $('#thresholdCopy').textContent=`낮음 < ${low}% · 높음 ≥ ${high}%`;
  const ring=$('#riskRing');ring.style.setProperty('--progress',`${Math.min(100,r.percentage)*3.6}deg`);
  ring.style.setProperty('--green',r.tier==='high'?'#a33a35':r.tier==='attention'?'#b56e16':'#155f4b');
  $('#qualityText').textContent=r.input_quality.optional_missing?`선택검사 ${r.input_quality.optional_missing}개 보완`:'모든 항목 입력됨';
  const max=Math.max(...r.factors.map(x=>Math.abs(x.contribution)),.01);
  $('#factorList').innerHTML=r.factors.map(f=>`<div class="factor ${f.contribution<0?'negative':''}"><div class="factor-head"><strong>${esc(f.label)}</strong><span>${esc(f.direction)}</span></div><div class="factor-bar"><i style="width:${Math.max(8,Math.abs(f.contribution)/max*100).toFixed(0)}%"></i></div></div>`).join('');
  const warning=r.warnings.length?r.warnings:['추가 경고는 없지만 결과는 의료진과 함께 검토해야 합니다.'];
  $('#warningList').innerHTML=warning.map(x=>`<li>${esc(x)}</li>`).join('');
  $('#warningSection').hidden=false;$('#resultDisclaimer').textContent=r.disclaimer;
  const emergency=$('#emergencyResult');emergency.hidden=!r.emergency;emergency.textContent=r.emergency?'급성 경고 증상이 있습니다. 결과와 관계없이 즉시 119 또는 응급의료기관에 연락하세요.':'';
  if(window.innerWidth<900) $('#resultCard').scrollIntoView({behavior:'smooth',block:'start'});
}
function resetResult(){state.result=null;$('#resultContent').hidden=true;$('#resultEmpty').hidden=false;}
function downloadCsv(){
  if(!state.result)return;const r=state.result;const rows=[['항목','값'],['분류단계',r.tier_ko],['상대적 가능성',r.percentage+'%'],['모델 버전',r.model_version],['주의','진단 또는 미래 절대위험이 아님']];
  Object.entries(r.normalized_input).forEach(([k,v])=>rows.push([k,v??'미입력']));r.factors.forEach((f,i)=>rows.push([`기여요인 ${i+1}`,`${f.label} (${f.direction})`]));
  const csv='\ufeff'+rows.map(row=>row.map(v=>'"'+String(v).replaceAll('"','""')+'"').join(',')).join('\r\n');
  const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download=`heartlens_result_${new Date().toISOString().slice(0,10)}.csv`;a.click();URL.revokeObjectURL(url);
}
document.addEventListener('DOMContentLoaded',init);
