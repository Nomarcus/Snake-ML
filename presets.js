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

  const PPO_PRESET={
    id:'ppo_7day_extreme',
    label:'PPO 7-dagars Extreme (Snake)',
    rewardConfig:{
      fruit:3.0,
      death:-6.0,
      stepCost:-0.01,
      potentialCoeff:0.6,
      newCellBonus:0.05,
      loopPenalty:-0.3,
      loopPenaltyEscalation:1.5,
      tightSpacePenalty:-0.4,
      wallHugPenalty:-0.02,
      tailPathBonus:0.1,
      clipMin:-2.5,
      clipMax:2.5,
      extremeFactor:1.0,
    },
    ppoHyper:{
      gamma:0.995,
      gaeLambda:0.95,
      clipRange:0.20,
      entropyCoeff:0.02,
      vfCoeff:0.5,
      maxGradNorm:0.5,
      rolloutSteps:4096,
      minibatchSize:512,
      epochsPerUpdate:6,
      learningRate:3e-4,
      learningRateSchedule:'cosine',
      obsNorm:true,
      rewardNorm:true,
      orthogonalInit:true,
      activation:'tanh',
      lstm:true,
      targetKL:0.01,
      temperature:1.2,
    },
    schedules:[
      {key:'entropyCoeff',target:0.001,updates:200},
      {key:'clipRange',target:0.10,updates:200},
      {key:'temperature',target:0.9,updates:200},
      {key:'learningRate',schedule:'cosine',warmupUpdates:10},
    ],
    curriculum:{
      transitions:[
        {from:'10x10',to:'15x15',condition:{avgFruits:4,window:200}},
        {from:'15x15',to:'20x20',condition:{avgFruits:6,window:200}},
      ],
      multiFruitAfter:'15x15',
    },
    plan:{
      duration:{value:7,unit:'days'},
      totalUpdates:200,
      milestones:normaliseMilestones([
        {
          fraction:0.0,
          label:'Initiera extreme factor',
          apply(runtime){
            if(!runtime.rewardConfig) runtime.rewardConfig={};
            runtime.rewardConfig.extremeFactor=1.0;
          },
        },
        {
          fraction:0.17,
          label:'Öka extreme factor',
          apply(runtime){
            if(!runtime.rewardConfig) runtime.rewardConfig={};
            const max=runtime.plan?.stagnation?.maxExtreme??2.0;
            runtime.rewardConfig.extremeFactor=Math.min(max,1.5);
          },
        },
        {
          fraction:0.33,
          label:'Stabilisera exploration',
          apply(runtime){
            if(!runtime.rewardConfig) runtime.rewardConfig={};
            if(!runtime.ppoHyper) runtime.ppoHyper={};
            runtime.ppoHyper.clipRange=0.15;
            runtime.ppoHyper.entropyCoeff=0.015;
            runtime.ppoHyper.temperature=1.05;
            runtime.rewardConfig.newCellBonus=0.03;
          },
        },
        {
          fraction:0.5,
          label:'Aktivera teacher assist',
          apply(runtime){
            runtime.teacherAssist={enabled:true,beta:0.1};
          },
        },
        {
          fraction:0.83,
          label:'Finjustera KL-mål',
          apply(runtime){
            if(!runtime.ppoHyper) runtime.ppoHyper={};
            runtime.ppoHyper.targetKL=0.01;
          },
        },
        {
          fraction:1.0,
          label:'Avsluta och utvärdera',
          apply(runtime){
            if(!runtime.plan) runtime.plan={};
            runtime.plan.evalNow=true;
          },
        },
      ]),
      stagnation:{
        windowEpisodes:10000,
        minDeltaFruitPerEp:0.05,
        bumpExtremeBy:0.25,
        maxExtreme:2.0,
      },
    },
  };

  global.SNAKE_PRESETS={...existingPresets,ppo_7day_extreme:PPO_PRESET};

  function ensureRuntime(runtime){
    const target=runtime||{};
    target.rewardConfig=target.rewardConfig?{...target.rewardConfig}:{ };
    target.ppoHyper=target.ppoHyper?{...target.ppoHyper}:{ };
    target.schedules=Array.isArray(target.schedules)?target.schedules.slice():[];
    target.curriculum=target.curriculum?{...target.curriculum}:{ };
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
    plan.milestones.forEach((milestone)=>{
      if(!milestone) return;
      const id=milestone.id??milestone.label;
      if(triggered.has(id)) return;
      if(fraction>=milestone.fraction){
        if(typeof milestone.apply==='function'){
          milestone.apply(runtime);
        }
        triggered.add(id);
        events.push({
          id,
          label:milestone.label,
          fraction:milestone.fraction,
          reason,
        });
      }
    });
    return events;
  }

  function sanitisePlan(plan){
    if(!plan) return null;
    const {triggered,_stagnationTracker,...rest}=plan;
    const copy={...rest};
    if(triggered instanceof Set){
      copy.triggered=Array.from(triggered);
    }else if(Array.isArray(triggered)){
      copy.triggered=triggered.slice();
    }
    return copy;
  }

  global.getSnakePreset=function getSnakePreset(id){
    const preset=global.SNAKE_PRESETS?.[id];
    return preset?deepClone(preset):null;
  };

  global.applySnakePreset=function applySnakePreset(runtime,id){
    const preset=global.SNAKE_PRESETS?.[id];
    if(!preset) return runtime||null;
    const target=ensureRuntime(runtime||{});
    target.activePreset=id;
    target.label=preset.label;
    target.rewardConfig={...deepClone(preset.rewardConfig)};
    target.ppoHyper={...deepClone(preset.ppoHyper)};
    target.schedules=deepClone(preset.schedules);
    target.curriculum=deepClone(preset.curriculum);
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

  global.applyPlanMilestones=function applyPlanMilestones(runtime,updateIndex){
    if(!runtime||!runtime.plan) return [];
    const plan=runtime.plan;
    const start=plan.startUpdateIndex??0;
    const total=Math.max(1,Number(plan.totalUpdates)||1);
    const progress=Math.max(0,(Number(updateIndex)||0)-start);
    const fraction=Math.max(0,Math.min(1,progress/total));
    plan.progressByUpdates=fraction;
    return evaluateMilestones(runtime,fraction,'updates');
  };

  global.applyPlanByTime=function applyPlanByTime(runtime){
    if(!runtime||!runtime.plan) return [];
    const plan=runtime.plan;
    const start=plan.startTimestamp;
    const totalMs=plan.durationMs??durationToMs(plan.duration);
    if(!start||!totalMs) return [];
    const elapsed=Math.max(0,Date.now()-start);
    const fraction=Math.max(0,Math.min(1,elapsed/totalMs));
    plan.progressByTime=fraction;
    return evaluateMilestones(runtime,fraction,'time');
  };

  global.maybeBumpExtremeOnStagnation=function maybeBumpExtremeOnStagnation(runtime,telemetry){
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
    if(!runtime.rewardConfig) runtime.rewardConfig={};
    const current=Number(runtime.rewardConfig.extremeFactor??1);
    const bump=Number(config.bumpExtremeBy??0);
    if(bump<=0) return null;
    const max=Number(config.maxExtreme??current);
    const next=Math.min(max,current+bump);
    if(next<=current) return null;
    runtime.rewardConfig.extremeFactor=next;
    return {changed:true,extremeFactor:next};
  };

  global.serializePlanForPrompt=function serializePlanForPrompt(plan){
    return sanitisePlan(plan);
  };
})(window);
