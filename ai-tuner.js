const API_URL='https://api.openai.com/v1/chat/completions';
const SYSTEM_PROMPT=`You are an expert reinforcement-learning coach for the classic game Snake.
The agent plays Snake on a 2-D grid where it collects fruit and grows longer.
The telemetry you receive describes recent episodes, current reward parameters, and performance trends.
Your job is to:

Evaluate the agent’s long-term progress and stability.

Suggest specific numeric adjustments to reward settings and key hyperparameters that will increase the chance of consistently reaching the maximum score without overfitting.

Explain your reasoning in 1–2 short paragraphs so a developer can follow your thought process.
Always respond with valid JSON containing:

{
  "rewardConfig": {...},
  "hyper": {...},
  "analysisText": "clear explanation of trends and adjustments"
}

Do not remove all rewards or penalties unless you clearly explain why that is optimal.`;

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

function wordCount(text){
  if(typeof text!=='string') return 0;
  const trimmed=text.trim();
  if(!trimmed) return 0;
  return trimmed.split(/\s+/).filter(Boolean).length;
}

function limitWords(text,maxWords){
  if(typeof text!=='string') return '';
  const trimmed=text.trim();
  if(!trimmed) return '';
  const parts=trimmed.split(/\s+/).filter(Boolean);
  if(parts.length<=maxWords) return parts.join(' ');
  return parts.slice(0,maxWords).join(' ');
}

function ensureSentence(text){
  if(typeof text!=='string') return '';
  const trimmed=text.trim();
  if(!trimmed) return '';
  const punctuation=/[.!?]$/.test(trimmed)?'':'';
  return `${trimmed}${punctuation}`;
}

function describeTrend(value,positiveWord='rising',negativeWord='falling',zeroWord='flat'){
  if(!Number.isFinite(value)||Math.abs(value)<1e-6) return zeroWord;
  const magnitude=Math.abs(value).toFixed(2);
  return value>0?`${positiveWord} by ${magnitude}`:`${negativeWord} by ${magnitude}`;
}

const CRASH_LABELS={
  wall:'wall hits',
  self:'self collisions',
  timeout:'timeouts',
  enemy:'enemy collisions',
  none:'no crash',
};

function humanizeCrash(key){
  if(!key) return '';
  return CRASH_LABELS[key]||key.replace(/_/g,' ');
}

function selectDominantCrash(crashCounts={},totalEpisodes=0){
  const entries=Object.entries(crashCounts).filter(([,count])=>Number.isFinite(count)&&count>0);
  if(!entries.length) return null;
  entries.sort((a,b)=>b[1]-a[1]);
  const [key,count]=entries[0];
  const share=totalEpisodes>0?Math.round((count/totalEpisodes)*100):null;
  return {key,count,share};
}

function summariseChangeList(changes=[],maxItems=3){
  if(!Array.isArray(changes)||!changes.length) return [];
  const entries=changes.slice(0,maxItems).map(change=>{
    const key=change.key??change.id??'value';
    const before=formatNumber(change.oldValue);
    const after=formatNumber(change.newValue);
    return `${key} ${before}→${after}`;
  });
  if(changes.length>maxItems&&entries.length){
    const lastIndex=entries.length-1;
    entries[lastIndex]=`${entries[lastIndex]} (+${changes.length-maxItems} more)`;
  }
  return entries;
}

function extractReasoningSnippet(response){
  if(!response||typeof response!=='object') return '';
  const candidates=[
    response.reasoning,
    response.analysis?.justeringar,
    response.analysis?.reasoning,
    response.analysis?.summary,
    Array.isArray(response.analysis?.key_findings)?response.analysis.key_findings.join(' '):'',
  ];
  for(const candidate of candidates){
    if(typeof candidate==='string'&&candidate.trim()){
      return limitWords(candidate,40);
    }
  }
  return '';
}

function buildAnalysisParagraph({telemetry,response,rewardChanges=[],hyperChanges=[]}){
  if(!telemetry||typeof telemetry!=='object'||!response) return '';
  const meta=telemetry.meta||{};
  const stats=telemetry.stats||{};
  const interval=Number(meta.interval)||0;
  const episode=Number(meta.episode)||0;
  const startEpisode=interval&&episode?Math.max(1,episode-interval+1):null;
  const spanText=interval&&episode?`Last ${interval} episodes (${startEpisode}–${episode})`:'Recent episodes';
  const rewardAvgText=Number.isFinite(stats.rewardAvg)?stats.rewardAvg.toFixed(2):'n/a';
  const rewardTrendText=describeTrend(stats.rewardTrend,'rising','falling','flat');
  const fruitTrendDescriptor=Number.isFinite(stats.fruitTrend)?describeTrend(stats.fruitTrend,'growing','shrinking','flat'):'flat';
  const bestLen=Number.isFinite(meta.best)?meta.best:null;

  const observationSentence=ensureSentence(
    `${spanText} show average reward ${rewardAvgText} with a ${rewardTrendText} trend, while fruit momentum is ${fruitTrendDescriptor}`+
    (bestLen?` and best length reached ${bestLen}`:'')
  );

  const stepsAvg=Number.isFinite(stats.stepsAvg)?Math.round(stats.stepsAvg):null;
  const loopsAvg=Number.isFinite(stats.loopsAvg)?stats.loopsAvg.toFixed(2):null;
  const fruitRate=Number.isFinite(stats.fruitRate)?stats.fruitRate.toFixed(3):null;
  const dominantCrash=selectDominantCrash(telemetry.crash,interval||telemetry.meta?.interval||rewardChanges.length+hyperChanges.length);
  const crashText=dominantCrash&&dominantCrash.share!==null?`${humanizeCrash(dominantCrash.key)} at ${dominantCrash.share}%`:(dominantCrash?humanizeCrash(dominantCrash.key):'');
  const trendParts=[];
  if(stepsAvg!==null) trendParts.push(`mean steps ${stepsAvg}`);
  if(loopsAvg!==null) trendParts.push(`loop hits ${loopsAvg}`);
  if(fruitRate!==null) trendParts.push(`fruit rate ${fruitRate} per step`);
  if(crashText) trendParts.push(`dominant exit ${crashText}`);
  const trendsSentence=trendParts.length?ensureSentence(`Telemetry also notes ${trendParts.join(', ')}.`):'';

  const rewardDescriptions=summariseChangeList(rewardChanges);
  const hyperDescriptions=summariseChangeList(hyperChanges);
  const adjustmentParts=[];
  if(rewardDescriptions.length) adjustmentParts.push(`reward tweaks ${rewardDescriptions.join('; ')}`);
  if(hyperDescriptions.length) adjustmentParts.push(`hyperparameter updates ${hyperDescriptions.join('; ')}`);
  const adjustmentsSentence=adjustmentParts.length?ensureSentence(`Adjustments apply ${adjustmentParts.join(' and ')}.`):'';

  const reasoningSnippet=extractReasoningSnippet(response);
  const reasoningSentence=reasoningSnippet?ensureSentence(`Rationale: ${reasoningSnippet}`):'';

  const statusMap={
    good:'is going well',
    stable:'is stable',
    bad:'needs improvement',
    uncertain:'needs closer monitoring',
  };
  const statusKey=response.assessment?.status;
  const defaultStatus=stats.rewardTrend>0.01?'is going well':stats.rewardTrend<-0.01?'needs improvement':'is stable';
  const statusText=statusMap[statusKey]||defaultStatus;
  const trendWord=response.assessment?.trend;
  const confidence=Number.isFinite(response.assessment?.confidence)?Math.round(response.assessment.confidence*100):null;
  const assessmentParts=[`Overall performance ${statusText}`];
  if(trendWord) assessmentParts.push(`trend looks ${trendWord}`);
  if(confidence!==null) assessmentParts.push(`confidence ${confidence}%`);
  const assessmentSentence=ensureSentence(assessmentParts.join(', '));

  const sentences=[observationSentence,trendsSentence,adjustmentsSentence,reasoningSentence,assessmentSentence].filter(Boolean);
  let text=sentences.join(' ');
  let totalWords=wordCount(text);

  if(totalWords<80){
    const extras=[];
    if(Number.isFinite(stats.timeToFruit)) extras.push(`mean time-to-fruit ${stats.timeToFruit.toFixed(1)} moves`);
    if(Number.isFinite(meta.envs)&&meta.envs>0) extras.push(`${meta.envs} environments active`);
    const breakdown=telemetry.rewardBreakdown;
    if(breakdown&&typeof breakdown==='object'){
      const entries=Object.entries(breakdown)
        .filter(([,value])=>Number.isFinite(value))
        .sort((a,b)=>Math.abs(b[1])-Math.abs(a[1]));
      if(entries.length){
        const [key,value]=entries[0];
        extras.push(`reward share led by ${key} at ${value.toFixed(2)}`);
      }
    }
    if(extras.length){
      text=`${text} ${ensureSentence(`Additional context: ${extras.join(', ')}.`)}`;
      totalWords=wordCount(text);
    }
  }

  if(totalWords>120&&reasoningSentence){
    const trimmedReason=ensureSentence(`Rationale: ${limitWords(reasoningSnippet,25)}`);
    const idx=sentences.indexOf(reasoningSentence);
    if(idx>=0){
      sentences[idx]=trimmedReason;
      text=sentences.filter(Boolean).join(' ');
      totalWords=wordCount(text);
    }
  }

  if(totalWords>120){
    text=limitWords(text,120);
  }

  return text.trim();
}

export function createAITuner(options={}){
  const {
    getVecEnv=()=>null,
    fetchTelemetry,
    applyRewardConfig,
    applyHyperparameters,
    log,
    apiKey=null,
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
  let lastAnalysisText='';

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
    const hasUpdates=((rewardResult?.changes?.length)||0)+((hyperResult?.changes?.length)||0)>0;
    if(hasUpdates){
      const analysisText=buildAnalysisParagraph({
        telemetry,
        response:parsed,
        rewardChanges:rewardResult?.changes||[],
        hyperChanges:hyperResult?.changes||[],
      });
      if(analysisText){
        lastAnalysisText=analysisText;
        console.log('[AI Analysis]',analysisText);
        logEvent({title:'AI analys',detail:analysisText,tone:'summary',episodeNumber:episode});
      }
    }
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
    getLastAnalysis(){ return lastAnalysisText; },
  };
}
