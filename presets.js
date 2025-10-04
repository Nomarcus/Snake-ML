(function(global){
  const existingPresets = global.SNAKE_PRESETS || {};

  function deepClone(value){
    if(Array.isArray(value)) return value.map(deepClone);
    if(value && typeof value === 'object'){
      const cloned={};
      for(const key of Object.keys(value)) cloned[key]=deepClone(value[key]);
      return cloned;
    }
    return value;
  }

  function durationToMs(duration){
    if(!duration||typeof duration!=='object') return 0;
    const value=Math.max(0,Number(duration.value)||0);
    const unit=(duration.unit||'days').toLowerCase();
    const unitMs={days:86400000,hours:3600000,minutes:60000};
    return value*(unitMs[unit]??86400000);
  }

  function normaliseMilestones(milestones){
    return (Array.isArray(milestones)?milestones:[]).map((item,index)=>({
      ...item,
      id:item.id??`milestone_${index}`,
      label:item.label??`Milestone ${Math.round((item.fraction??0)*100)}%`,
      fraction:Number.isFinite(item.fraction)?Math.max(0,Math.min(1,item.fraction)):0,
    }));
  }

  // === 🟢 PPO STAGE 1: STABLE EXPLORER ===
  const PPO_STAGE1 = {
    id: 'ppo_stage1_stable_explorer',
    label: 'PPO – Stage 1 (Stable Explorer)',
    rewardConfig: {
      fruit: 1.2,
      death: -6.0,
      stepCost: -0.01,
      potentialCoeff: 0.6,
      newCellBonus: 0.05,
      loopPenalty: -0.25,
      loopPenaltyEscalation: 1.4,
      tightSpacePenalty: -0.3,
      wallHugPenalty: -0.02,
      tailPathBonus: 0.08,
      openSpaceBonus: 2.5,
      compactnessBonus: 0.25,
      timeoutPenalty: -10.0,
      extremeFactor: 1.0,
    },
    ppoHyper: {
      gamma: 0.995,
      gaeLambda: 0.95,
      clipRange: 0.15,
      entropyCoeff: 0.015,
      vfCoeff: 0.5,
      maxGradNorm: 0.5,
      rolloutSteps: 2048,
      minibatchSize: 512,
      epochsPerUpdate: 10,
      learningRate: 3e-4,
      learningRateSchedule: 'cosine',
      obsNorm: true,
      rewardNorm: true,
      orthogonalInit: true,
      activation: 'tanh',
      lstm: true,
      targetKL: 0.01,
      temperature: 1.1,
    },
    schedules: [
      { key: 'entropyCoeff', target: 0.005, updates: 150 },
      { key: 'clipRange', target: 0.1, updates: 150 },
      { key: 'temperature', target: 0.9, updates: 150 },
      { key: 'learningRate', schedule: 'cosine', warmupUpdates: 10 },
    ],
    curriculum: {
      transitions: [
        { from: '10x10', to: '15x15', condition: { avgFruits: 4, window: 200 } },
        { from: '15x15', to: '20x20', condition: { avgFruits: 7, window: 200 } },
      ],
      multiFruitAfter: '15x15',
    },
    plan: {
      duration: { value: 7, unit: 'days' },
      totalUpdates: 200,
      milestones: normaliseMilestones([
        { fraction: 0.0, label: 'Initiera stabil PPO-session', apply(rt){rt.rewardConfig.extremeFactor=1.0;} },
        { fraction: 0.25, label: 'Stabilisera utforskning', apply(rt){rt.ppoHyper.entropyCoeff=0.012;rt.ppoHyper.clipRange=0.15;} },
        { fraction: 0.5, label: 'Finjustera belöningsbalans', apply(rt){rt.rewardConfig.openSpaceBonus=3.0;rt.rewardConfig.compactnessBonus=0.3;rt.rewardConfig.timeoutPenalty=-12.0;} },
        { fraction: 0.83, label: 'Aktivera teacher assist', apply(rt){rt.teacherAssist={enabled:true,beta:0.1};} },
        { fraction: 1.0, label: 'Avsluta och utvärdera', apply(rt){rt.plan.evalNow=true;} },
      ]),
      stagnation:{windowEpisodes:8000,minDeltaFruitPerEp:0.05,bumpExtremeBy:0.2,maxExtreme:1.8},
    },
  };

  // === 🟠 PPO STAGE 2: ADAPTIVE GROWTH ===
  const PPO_STAGE2 = {
    id: 'ppo_stage2_adaptive_growth',
    label: 'PPO – Stage 2 (Adaptive Growth)',
    rewardConfig: {
      fruit: 1.5,
      death: -6.5,
      stepCost: -0.012,
      newCellBonus: 0.07,
      loopPenalty: -0.3,
      trapPenalty: -0.5,
      openSpaceBonus: 2.0,
      compactnessBonus: 0.25,
      timeoutPenalty: -12.0,
      extremeFactor: 1.2,
    },
    ppoHyper: {
      gamma: 0.996,
      gaeLambda: 0.96,
      clipRange: 0.18,
      entropyCoeff: 0.012,
      vfCoeff: 0.5,
      rolloutSteps: 3072,
      minibatchSize: 512,
      epochsPerUpdate: 12,
      learningRate: 2.5e-4,
      targetKL: 0.012,
      temperature: 1.05,
    },
    plan: {
      duration: { value: 7, unit: 'days' },
      totalUpdates: 200,
      milestones: normaliseMilestones([
        { fraction: 0.0, label: 'Initiera adaptiv träning', apply(rt){rt.rewardConfig.extremeFactor=1.2;} },
        { fraction: 0.33, label: 'Skruva upp utforskning', apply(rt){rt.ppoHyper.entropyCoeff=0.018;} },
        { fraction: 0.66, label: 'Balansera värden', apply(rt){rt.rewardConfig.openSpaceBonus=3.0;rt.rewardConfig.trapPenalty=-0.8;} },
        { fraction: 1.0, label: 'Aktivera teacher assist & eval', apply(rt){rt.teacherAssist={enabled:true,beta:0.1};rt.plan.evalNow=true;} },
      ]),
      stagnation:{windowEpisodes:10000,minDeltaFruitPerEp:0.04,bumpExtremeBy:0.25,maxExtreme:2.0},
    },
  };

  // === 🔴 PPO STAGE 3: PRECISION FINISHER ===
  const PPO_STAGE3 = {
    id: 'ppo_stage3_precision_finisher',
    label: 'PPO – Stage 3 (Precision Finisher)',
    rewardConfig: {
      fruit: 1.0,
      death: -7.0,
      stepCost: -0.009,
      openSpaceBonus: 3.5,
      compactnessBonus: 0.4,
      timeoutPenalty: -15.0,
      loopPenalty: -0.4,
      trapPenalty: -0.7,
      extremeFactor: 1.5,
    },
    ppoHyper: {
      gamma: 0.997,
      gaeLambda: 0.97,
      clipRange: 0.12,
      entropyCoeff: 0.008,
      vfCoeff: 0.55,
      rolloutSteps: 4096,
      minibatchSize: 512,
      epochsPerUpdate: 14,
      learningRate: 2e-4,
      targetKL: 0.009,
      temperature: 0.95,
    },
    plan: {
      duration: { value: 10, unit: 'days' },
      totalUpdates: 250,
      milestones: normaliseMilestones([
        { fraction: 0.0, label: 'Initiera precisionsträning', apply(rt){rt.rewardConfig.extremeFactor=1.5;} },
        { fraction: 0.25, label: 'Finjustera policy-klippning', apply(rt){rt.ppoHyper.clipRange=0.1;} },
        { fraction: 0.5, label: 'Sänk exploration & öka stabilitet', apply(rt){rt.ppoHyper.entropyCoeff=0.006;rt.ppoHyper.temperature=0.9;} },
        { fraction: 0.8, label: 'Förbättra värdefunktion', apply(rt){rt.ppoHyper.vfCoeff=0.6;} },
        { fraction: 1.0, label: 'Avsluta och utvärdera', apply(rt){rt.plan.evalNow=true;} },
      ]),
      stagnation:{windowEpisodes:12000,minDeltaFruitPerEp:0.03,bumpExtremeBy:0.1,maxExtreme:1.7},
    },
  };

  // === REGISTER ALL PRESETS ===
  global.SNAKE_PRESETS = {
    ...existingPresets,
    ppo_stage1_stable_explorer: PPO_STAGE1,
    ppo_stage2_adaptive_growth: PPO_STAGE2,
    ppo_stage3_precision_finisher: PPO_STAGE3,
  };

  // === RUNTIME / PLAN LOGIC (unchanged) ===
  function ensureRuntime(runtime){
    const target=runtime||{};
    target.rewardConfig={...target.rewardConfig};
    target.ppoHyper={...target.ppoHyper};
    target.schedules=Array.isArray(target.schedules)?target.schedules.slice():[];
    target.curriculum={...target.curriculum};
    target.progress=target.progress||{updatesCompleted:0};
    target.plan=target.plan||{};
    if(!(target.plan.triggered instanceof Set)){
      target.plan.triggered=new Set(Array.isArray(target.plan.triggered)?target.plan.triggered:[]);
    }
    return target;
  }

  function evaluateMilestones(runtime,fraction,reason){
    const plan=runtime?.plan;
    if(!plan||!Array.isArray(plan.milestones)) return [];
    const triggered=plan.triggered instanceof Set?plan.triggered:new Set();
    plan.triggered=triggered;
    const events=[];
    plan.milestones.forEach(m=>{
      if(!m) return;
      const id=m.id??m.label;
      if(triggered.has(id)) return;
      if(fraction>=m.fraction){
        if(typeof m.apply==='function') m.apply(runtime);
        triggered.add(id);
        events.push({id,label:m.label,fraction:m.fraction,reason});
      }
    });
    return events;
  }

  function sanitisePlan(plan){
    if(!plan) return null;
    const {triggered,_stagnationTracker,...rest}=plan;
    const copy={...rest};
    if(triggered instanceof Set) copy.triggered=Array.from(triggered);
    else if(Array.isArray(triggered)) copy.triggered=triggered.slice();
    return copy;
  }

  global.getSnakePreset=function(id){
    const preset=global.SNAKE_PRESETS?.[id];
    return preset?deepClone(preset):null;
  };

  global.applySnakePreset=function(runtime,id){
    const preset=global.SNAKE_PRESETS?.[id];
    if(!preset) return runtime||null;
    const target=ensureRuntime(runtime||{});
    target.activePreset=id;
    target.label=preset.label;
    target.rewardConfig=deepClone(preset.rewardConfig);
    target.ppoHyper=deepClone(preset.ppoHyper);
    target.schedules=deepClone(preset.schedules||[]);
    target.curriculum=deepClone(preset.curriculum||{});
    const plan=preset.plan?deepClone(preset.plan):{};
    plan.milestones=normaliseMilestones(plan.milestones);
    plan.triggered=new Set();
    plan.totalUpdates=plan.totalUpdates??200;
    plan.duration=plan.duration||{value:7,unit:'days'};
    plan.durationMs=durationToMs(plan.duration);
    plan.startTimestamp=Date.now();
    plan.startUpdateIndex=target.progress?.updatesCompleted??0;
    plan.evalNow=false;
    plan.progressByUpdates=0;
    plan.progressByTime=0;
    plan._stagnationTracker=null;
    target.plan=plan;
    target.teacherAssist={enabled:false,beta:0};
    return target;
  };

  global.applyPlanMilestones=function(runtime,updateIndex){
    if(!runtime||!runtime.plan) return [];
    const plan=runtime.plan;
    const start=plan.startUpdateIndex??0;
    const total=Math.max(1,Number(plan.totalUpdates)||1);
    const progress=Math.max(0,(Number(updateIndex)||0)-start);
    const fraction=Math.max(0,Math.min(1,progress/total));
    plan.progressByUpdates=fraction;
    return evaluateMilestones(runtime,fraction,'updates');
  };

  global.applyPlanByTime=function(runtime,telemetry){
    if(!runtime||!runtime.plan) return [];
    const plan=runtime.plan;
    if(!plan.startTimestamp){
      const hintedStart=Number(telemetry?.planStartMs);
      plan.startTimestamp=Number.isFinite(hintedStart)?hintedStart:Date.now();
    }
    const start=plan.startTimestamp;
    const totalMs=plan.durationMs??durationToMs(plan.duration);
    if(!start||!totalMs) return [];
    const nowCandidate=Number(telemetry?.nowMs);
    const now=Number.isFinite(nowCandidate)?nowCandidate:Date.now();
    const elapsed=Math.max(0,now-start);
    const fraction=Math.max(0,Math.min(1,elapsed/totalMs));
    plan.progressByTime=fraction;
    return evaluateMilestones(runtime,fraction,'time');
  };

  global.maybeBumpExtremeOnStagnation=function(runtime,telemetry){
    if(!runtime||!runtime.plan||!runtime.plan.stagnation) return null;
    const config=runtime.plan.stagnation;
    const episodes=Number(telemetry?.episodesCompleted??0);
    const fruitRate=Number(telemetry?.fruitsPerEpisodeRolling);
    if(!Number.isFinite(episodes)||!Number.isFinite(fruitRate)) return null;
    if(!runtime.plan._stagnationTracker){
      runtime.plan._stagnationTracker={episode:episodes,fruit:fruitRate};
      return null;
    }
    const tracker=runtime.plan._stagnationTracker;
    const windowEpisodes=Math.max(1,Number(config.windowEpisodes)||1);
    if(episodes-tracker.episode<windowEpisodes) return null;
    const delta=fruitRate-tracker.fruit;
    runtime.plan._stagnationTracker={episode:episodes,fruit:fruitRate};
    if(delta>=Number(config.minDeltaFruitPerEp??0)) return null;
    const current=Number(runtime.rewardConfig.extremeFactor??1);
    const bump=Number(config.bumpExtremeBy??0);
    if(bump<=0) return null;
    const max=Number(config.maxExtreme??current);
    const next=Math.min(max,current+bump);
    if(next<=current) return null;
    runtime.rewardConfig.extremeFactor=next;
    return {changed:true,extremeFactor:next};
  };

  global.serializePlanForPrompt=function(plan){
    return sanitisePlan(plan);
  };
})(window);
