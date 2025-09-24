const API_URL='https://api.openai.com/v1/chat/completions';
const SYSTEM_PROMPT=`Du är en expert på reinforcement learning.
Ditt mål är att justera Snake-MLs belöningsparametrar och centrala
hyperparametrar så att ormen klarar spelet konsekvent.
Returnera ENDAST minifierad JSON med nya värden för alla parametrar
du vill uppdatera, t.ex.
{
  "rewardConfig": {stepPenalty:0.008, fruitReward:12, ...},
  "hyper": {gamma:0.985, lr:0.0004, epsDecay:90000, ...}
}`;

function resolveApiKey(preferred){
  if(typeof preferred==='string' && preferred.trim()) return preferred.trim();
  if(typeof globalThis!=='undefined' && typeof globalThis.__OPENAI_KEY==='string' && globalThis.__OPENAI_KEY.trim()){
    return globalThis.__OPENAI_KEY.trim();
  }
  if(typeof process!=='undefined' && process?.env?.OPENAI_API_KEY){
    return String(process.env.OPENAI_API_KEY).trim();
  }
  return null;
}

function formatNumber(value){
  if(value===null||value===undefined||Number.isNaN(value)) return '—';
  if(typeof value==='number'){
    const abs=Math.abs(value);
    if(abs>=100) return value.toFixed(0);
    if(abs>=10) return value.toFixed(1);
    if(abs>=1) return value.toFixed(2);
    return value.toFixed(3);
  }
  return String(value);
}

function formatChanges(changes=[]){
  if(!Array.isArray(changes) || !changes.length) return '';
  return changes.map(change=>{
    const key=change.key??change.id??'värde';
    const before=formatNumber(change.oldValue);
    const after=formatNumber(change.newValue);
    return `${key}: ${before}→${after}`;
  }).join(', ');
}

export function createAITuner(options={}){
  const {
    getVecEnv=()=>null,
    fetchTelemetry,
    applyRewardConfig,
    applyHyperparameters,
    log,
    apiKey:null,
  }=options;
  if(typeof fetchTelemetry!=='function'){
    throw new Error('createAITuner requires a fetchTelemetry() function');
  }
  const logger=typeof log==='function'?log:()=>{};
  const resolveEnv=typeof getVecEnv==='function'?getVecEnv:()=>getVecEnv??null;
  let enabled=false;
  let interval=1000;
  let busy=false;
  let warnedNoKey=false;
  let warnedNoFetch=false;

  function logEvent(payload){
    try{
      logger(payload);
    }catch(err){
      console.warn('[ai-tuner] failed to log event',err);
    }
  }

  async function runTuningCycle(telemetry,episode){
    if(typeof fetch!=='function'){
      if(!warnedNoFetch){
        logEvent({title:'AI Auto-Tune',detail:'fetch saknas i miljön – kan inte kontakta OpenAI.',tone:'error',episodeNumber:episode});
        warnedNoFetch=true;
      }
      return;
    }
    const key=resolveApiKey(apiKey);
    if(!key){
      if(!warnedNoKey){
        logEvent({title:'AI Auto-Tune',detail:'OPENAI_API_KEY saknas. Hoppar över justeringar.',tone:'error',episodeNumber:episode});
        warnedNoKey=true;
      }
      return;
    }
    const payload=JSON.stringify(telemetry);
    const response=await fetch(API_URL,{
      method:'POST',
      headers:{
        'Content-Type':'application/json',
        Authorization:`Bearer ${key}`,
      },
      body:JSON.stringify({
        model:'gpt-4o-mini',
        temperature:0.2,
        messages:[
          {role:'system',content:SYSTEM_PROMPT},
          {role:'user',content:payload},
        ],
        response_format:{type:'json_object'},
      }),
    });
    if(!response.ok){
      const text=await response.text();
      throw new Error(`OpenAI ${response.status}: ${text.slice(0,200)}`);
    }
    const data=await response.json();
    const content=data?.choices?.[0]?.message?.content;
    if(!content){
      throw new Error('Saknar innehåll i OpenAI-svar.');
    }
    let parsed;
    try{
      parsed=JSON.parse(content);
    }catch(err){
      throw new Error(`Kunde inte tolka JSON: ${content}`);
    }
    const rewardResult=typeof applyRewardConfig==='function'
      ?applyRewardConfig(parsed.rewardConfig||parsed.reward||{})
      :{changes:[],config:null};
    const hyperResult=typeof applyHyperparameters==='function'
      ?applyHyperparameters(parsed.hyper||parsed.hyperparameters||{})
      :{changes:[],hyper:null};
    const envInstance=resolveEnv?.();
    if(envInstance?.setRewardConfig && rewardResult?.config){
      try{
        envInstance.setRewardConfig(rewardResult.config);
      }catch(err){
        console.warn('[ai-tuner] setRewardConfig failed',err);
      }
    }
    const rewardSummary=formatChanges(rewardResult?.changes);
    const hyperSummary=formatChanges(hyperResult?.changes);
    if(rewardSummary){
      logEvent({title:'AI belöningar',detail:rewardSummary,tone:'reward',episodeNumber:episode});
    }
    if(hyperSummary){
      logEvent({title:'AI hyperparametrar',detail:hyperSummary,tone:'lr',episodeNumber:episode});
    }
    if(!rewardSummary && !hyperSummary){
      logEvent({title:'AI Auto-Tune',detail:'Ingen justering rekommenderades.',tone:'ai',episodeNumber:episode});
    }
  }

  function maybeTune({episode}={}){
    if(!enabled) return;
    if(!episode || episode%interval!==0) return;
    if(busy) return;
    busy=true;
    (async()=>{
      try{
        const telemetry=await Promise.resolve(fetchTelemetry({interval,episode}));
        if(!telemetry){
          return;
        }
        await runTuningCycle(telemetry,episode);
      }catch(err){
        console.error('[ai-tuner]',err);
        logEvent({title:'AI Auto-Tune',detail:err.message||String(err),tone:'error',episodeNumber:episode});
      }finally{
        busy=false;
      }
    })();
  }

  return {
    setEnabled(value){ enabled=!!value; },
    setInterval(value){
      const num=Number(value);
      if(Number.isFinite(num) && num>0){
        interval=Math.max(1,Math.floor(num));
      }
    },
    maybeTune,
    isEnabled(){ return enabled; },
    getInterval(){ return interval; },
  };
}
