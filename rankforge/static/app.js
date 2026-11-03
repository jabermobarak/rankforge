'use strict';
const $ = id => document.getElementById(id);
let current = null;
let bearer = '';
const percent = x => (100 * x).toFixed(1) + '%';
const palette = {Outdoors:['#dce7c9','#526a34','⌁'],Audio:['#e0def1','#72658e','◉'],Home:['#eddcc7','#8e6c49','⌂'],Fitness:['#d2e2db','#537464','◎'],Books:['#e9dfd0','#8d7057','▤'],Travel:['#d6e2e8','#58788b','↗'],Art:['#e8d9d9','#9d6d71','✳'],Tech:['#d9e0d3','#607352','⌘']};
function node(tag, cls, text) { const e = document.createElement(tag); if(cls)e.className=cls; if(text!==undefined)e.textContent=text; return e; }
function error(message) { $('error').hidden=!message; $('error').textContent=message || ''; }
async function api(path, options={}) {
  const headers = {...(options.headers || {})};
  if(bearer) headers.Authorization='Bearer '+bearer;
  const response=await fetch(path,{...options,headers});
  const data=await response.json();
  if(!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Request rejected. Check the input and API access settings.');
  return data;
}
async function history() {
  const data=await api('/api/history/'+encodeURIComponent($('user').value));
  $('history-summary').textContent=`${data.training_interactions} training interactions · ${data.live_feedback} saved picks`;
  $('profile').replaceChildren();
  data.categories.sort((a,b)=>b.share-a.share).slice(0,5).forEach(c=>{
    const row=node('div','profile-row'); const label=node('div','profile-label');
    label.append(node('span','',c.category),node('span','',percent(c.share)));
    const bar=node('div','bar');const fill=node('i');fill.style.width=percent(c.share);bar.append(fill);
    row.append(label,bar);$('profile').append(row);
  });
}
async function ledger() {
  const data=await api('/api/experiments'); $('ledger').replaceChildren();
  ['A','B'].forEach(variant=>{
    const v=data.variants.find(x=>x.variant===variant) || {requests:0,request_ctr:0};
    const card=node('div');card.append(node('span','',variant==='A'?'A / Popularity':'B / Neural ranking'),node('strong','',percent(v.request_ctr)),node('span','',`${v.requests} requests · request CTR`));$('ledger').append(card);
  });
  const total=node('div');total.append(node('span','','ATTRIBUTED FEEDBACK'),node('strong','',data.feedback_events),node('span','','deduplicated item saves'));$('ledger').append(total);
}
function renderFeed(data) {
  current=data; $('feed').replaceChildren();
  data.items.forEach(item=>{
    const card=node('article','item');const [bg,ink,glyph]=palette[item.category] || ['#e0e6d8','#64744d','◇'];
    const art=node('div','item-art');art.style.setProperty('--tile',bg);art.style.setProperty('--tile-ink',ink);art.setAttribute('aria-hidden','true');art.append(node('span','rank',String(item.rank).padStart(2,'0')),node('span','glyph',glyph));
    const save=node('button','save','Save +');save.setAttribute('aria-label','Save '+item.title);
    const exposure={request_id:data.request_id,user_id:data.user_id,item_id:item.item_id,event_id:crypto.randomUUID()};
    save.addEventListener('click',async()=>{
      save.disabled=true;
      try { await api('/api/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(exposure)});save.textContent='Saved ✓';save.classList.add('saved');await Promise.all([history(),ledger()]); }
      catch(e){save.disabled=false;error(e.message);}
    });
    const bottom=node('div','item-bottom');bottom.append(node('span','score',item.score.toFixed(3)+' affinity'),save);
    card.append(art,node('div','item-category',item.category),node('h3','',item.title),node('div','reason',item.reason),bottom);$('feed').append(card);
  });
  if(!data.items.length)$('feed').append(node('p','','No unseen eligible items remain for this profile.'));
  $('inference').textContent=`${data.method.replaceAll('_',' ')} · ${data.latency_ms.toFixed(1)} ms`;
  $('variant').textContent=data.variant==='A'?'A / Popularity baseline':'B / Neural retrieval + ranking';
}
async function refresh() {
  error('');$('refresh').disabled=true;$('refresh').textContent='Building feed…';
  try {
    const query=new URLSearchParams({user_id:$('user').value,k:'12',diversity:$('diversity').value});
    renderFeed(await api('/api/recommendations?'+query)); await Promise.all([history(),ledger()]);
  } catch(e){error(e.message);} finally{$('refresh').disabled=false;$('refresh').textContent='Generate feed ↗';}
}
async function init() {
  error('');
  try {
    const [report,users]=await Promise.all([api('/api/report'),api('/api/users')]);
    $('stat-items').textContent=report.sizes.items.toLocaleString();$('stat-events').textContent=report.sizes.train.toLocaleString();$('stat-recall').textContent=percent(report.test.neural.recall_at_10);$('stat-version').textContent=report.model_version.slice(0,8);
    $('user').replaceChildren(...users.users.map(u=>{const o=node('option','',u);o.value=u;return o;}));
    const cold=node('option','','New visitor / cold start');cold.value=users.cold_start_user;$('user').append(cold);
    $('evaluation-body').replaceChildren();
    const names={popular:'Popularity baseline',category:'Category baseline',neural:'Two-tower + profile ranking'};
    Object.entries(report.test).forEach(([method,m])=>{const row=node('tr',method==='neural'?'highlight':'');row.append(node('td','',names[method]),node('td','',percent(m.recall_at_10)),node('td','',m.ndcg_at_10.toFixed(3)),node('td','',percent(m.catalog_coverage_at_10)));$('evaluation-body').append(row);});
    $('eval-note').textContent=`${report.sizes.test.toLocaleString()} held-out interactions · ${report.test.neural.users} users · Candidate recall @80: ${percent(report.candidate_recall_at_80)} · Source: ${report.data_source}. Synthetic results demonstrate the pipeline, not real customer performance.`;
    await refresh();
  } catch(e){error(e.message+' If needed, start the engine with python -m rankforge.cli demo.');}
}
$('refresh').addEventListener('click',refresh);$('user').addEventListener('change',refresh);
$('diversity').addEventListener('input',()=>{$('div-value').textContent=Number($('diversity').value).toFixed(2);});
$('ledger-refresh').addEventListener('click',()=>ledger().catch(e=>error(e.message)));
$('connect').addEventListener('click',()=>{bearer=$('token').value;$('token').value='';init();});
init();
