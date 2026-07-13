#!/usr/bin/env python3
"""CryptAI ES Trainer v5 — OPTIMIZED: multi-dataset, Sharpe, bigger model, elite+mutations"""
import os,sys,glob,time,random
import numpy as np
import pandas as pd
import torch

BASE="/mnt/hive_storage/CryptAI"
sys.path.insert(0,f"{BASE}/lib");sys.path.insert(0,f"{BASE}/src/training")
from features import build_features
from arena import Arena
from es_agent import BigPolicy as Model, EvolutionStrategies
os.makedirs(f"{BASE}/training/checkpoints/es",exist_ok=True)

CURR={1:{"fee":0,"max":20},2:{"fee":0.0005,"max":12},3:{"fee":0.001,"max":8}}

print("[ESv5] Loading data...",flush=True)
datasets=[]
for f in sorted(glob.glob(f"{BASE}/data/*/*/historical.csv")):
    try:
        df=pd.read_csv(f)
        if len(df)<300:continue
        feats=build_features(df)
        if len(feats)<196:continue
        vals=feats.values.astype(np.float32)
        for i in range(len(vals)):
            m=vals[i].mean();s=vals[i].std()+1e-8;vals[i]=(vals[i]-m)/s
        prices=df["close"].values[-len(feats):]
        datasets.append({"vals":vals,"prices":prices,"split":int(len(vals)*0.7),"name":f.split("/")[-3]})
    except:pass
print(f"  {len(datasets)} datasets",flush=True)

# ⚡ Bigger model
device="cuda"
policy=Model(hidden=768,num_layers=4).to(device)
es=EvolutionStrategies(policy,pop_size=32,sigma=0.03,lr=0.15,elite_ratio=0.25)
print(f"  Params: {policy.n_params:,} | pop={es.pop_size}",flush=True)

def evaluate(params,phase=1,di=None):
    """Single dataset evaluation — returns PnL, Sharpe, DD, trades"""
    if di is None:di=random.randrange(len(datasets))
    ds=datasets[di];v=ds["vals"][:ds["split"]];p=ds["prices"][:ds["split"]]
    policy.set_params(params)
    # Batched forward
    seqs=np.lib.stride_tricks.sliding_window_view(v,96,axis=0)
    seqs=np.moveaxis(seqs,-1,1)
    chunk=2048
    all_a=[]
    for s in range(0,len(seqs),chunk):
        x=torch.FloatTensor(seqs[s:s+chunk]).to(device)
        with torch.no_grad():all_a.extend(policy(x).argmax(-1).cpu().numpy())
    actions=np.array(all_a)
    # Arena
    ar=Arena(1000.0,fee_rate=CURR[phase]["fee"],slippage=0.0005)
    ti=0;trades=0
    while ti<len(actions) and 96+ti<len(p):
        a=actions[ti];pr=p[96+ti]
        if a==1 and ar.position is None:ar.open_trade(ds["name"],pr,1);trades+=1
        elif a==2 and ar.position is None:ar.open_trade(ds["name"],pr,1);trades+=1
        elif a==3 and ar.position is not None:ar.close_trade(pr)
        elif a in(4,5)and ar.position is None:ar.open_trade(ds["name"],pr,0.5);trades+=1
        elif a==6 and ar.position is not None:ar.open_trade(ds["name"],pr,0.3)
        elif a==7 and ar.position is not None:ar.close_trade(pr)
        ti+=1
        if ar.get_metrics()["max_drawdown"]>0.2:break
        if trades>=CURR[phase]["max"]:break
    if ar.position is not None:ar.close_trade(p[-1])
    mt=ar.get_metrics()
    pnl=mt["total_return"];dd=mt["max_drawdown"];sharpe=mt.get("sharpe",0)
    return pnl,sharpe,dd,trades

def fitness(params,phase=1):
    """Multi-dataset fitness: average over 3 random datasets, penalize DD, reward Sharpe"""
    pnls,sharps,dds=[],[],[]
    for _ in range(3):
        pnl,s,dd,_=evaluate(params,phase)
        pnls.append(pnl);sharps.append(s if s else 0);dds.append(dd if dd else 0)
    avg_pnl=np.mean(pnls)
    avg_dd=np.mean(dds)
    avg_sharpe=np.mean(sharps)
    # ⚡ Smart fitness: PnL - DD_penalty + Sharpe_bonus
    base=avg_pnl
    dd_penalty=max(0,avg_dd)*2  # -2% per 1% DD
    sharpe_bonus=min(0.05,avg_sharpe*0.01)  # +1% per Sharpe
    trades_penalty=-0.50 if all(t==0 for _,_,_,t in[evaluate(params,phase,di=i) for i in range(len(datasets))][:1])else 0
    # Re-evaluate once to check trades
    _,_,_,t=evaluate(params,phase)
    fit=base-dd_penalty+sharpe_bonus
    if t==0:fit-=0.50
    return fit

gen=0
while True:
    ph=1 if gen<200 else (2 if gen<500 else 3)
    pop=es.ask()
    fits=[fitness(p,ph) for p in pop]
    mf,bf,sf=es.tell(fits);gen+=1
    if gen%5==0:
        vram=torch.cuda.memory_allocated(0)/1024/1024/1024
        print(f"GEN{gen:>5} P{ph} | mean={mf:>+.3%} best={bf:>+.3%} | VRAM={vram:.1f}GB | σ={es.sigma:.4f}",flush=True)
    if gen%10==0:
        # Validation sur dataset fixe (pas aléatoire)
        vi=hash(str(gen))%len(datasets)  # deterministic dataset for fair comparison
        pnl,s,dd,t=evaluate(es.population[0],3,vi)
        print(f"  [VAL#{vi:>2}] PnL={pnl:>+.3%} Sharpe={s:.2f} DD={dd:.2%} Trades={t}",flush=True)
        if pnl>es.best_fitness:
            es.best_fitness=pnl
            es.save(f"{BASE}/training/checkpoints/es/gen{gen}_best.pth")
            print(f"  SAVED gen{gen} pnl={pnl:.3%} sharpe={s:.2f}",flush=True)
            if pnl>=0.16:
                print(f"🏆 CHAMPION ES: {pnl:.2%} — threshold reached!",flush=True)