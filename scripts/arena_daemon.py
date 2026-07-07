#!/usr/bin/env python3
import os,sys,json,glob,time,torch,pandas as pd,numpy as np
BASE='/mnt/hive_storage/CryptAI'
sys.path.insert(0,f'{BASE}/lib');sys.path.insert(0,f'{BASE}')
from features import build_features
from models_v2 import DeepLSTM,DeepTransformer,TCN
from models import MLPDirection
from arena import Arena
from collections import defaultdict

def load_model(ckpt):
    d=torch.load(ckpt,map_location='cpu');cfg=d.get('config',{});n=d.get('feat_names',None)
    m=None;sl=cfg.get('sl',96);hd=cfg.get('hd',128);nl=cfg.get('nl',2)
    md=cfg.get('model','mlp')
    if 'transformer' in md:m=DeepTransformer(25,hd,8,nl,sl)
    elif 'tcn' in md:m=TCN(25,hd,nl,sl)
    elif 'lstm' in md:m=DeepLSTM(25,hd,nl,sl)
    else:m=MLPDirection(25*sl)
    m.load_state_dict(d['state']);m.eval()
    return m,cfg

def test_champion(model,cfg):
    arena=Arena(1000.0)
    sl=cfg.get('sl',96)
    for f in sorted(glob.glob(f'{BASE}/data/*/4h/historical.csv'))[:3]:
        try:
            df=pd.read_csv(f,parse_dates=['timestamp'])
            feats=build_features(df)
            if len(feats)<sl:continue
            vals=feats.values.astype(np.float32)
            for i in range(len(vals)):mn=vals[i].mean();s=vals[i].std()+1e-8;vals[i]=(vals[i]-mn)/s
            prices=df['close'].values[-len(feats):]
            for i in range(sl,len(vals)):
                x=torch.FloatTensor(vals[i-sl:i]).unsqueeze(0)
                with torch.no_grad():p=model(x).argmax(1).item()
                if p==2 and arena.position is None:arena.open_trade('ARENA',prices[i],1)
                elif p==0 and arena.position is not None:arena.close_trade(prices[i])
            if arena.position is not None:arena.close_trade(prices[-1])
        except:pass
    return arena.get_metrics()

while True:
    ckpts=sorted(glob.glob(f'{BASE}/training/checkpoints/*.pth'),key=os.path.getmtime,reverse=True)
    tested=set(os.listdir(f'{BASE}/arena/champions/'))
    for ckpt in ckpts[:5]:
        name=os.path.basename(ckpt).replace('.pth','')
        if name+'.json' in tested:continue
        try:
            model,cfg=load_model(ckpt)
            metrics=test_champion(model,cfg)
            with open(f'{BASE}/arena/champions/{name}.json','w') as f:
                json.dump({'config':cfg,'results':metrics,'ckpt':ckpt},f,indent=2)
            print(f'[arena] {name}: Return={metrics["total_return"]:.2%} DD={metrics["max_drawdown"]:.2%}',flush=True)
            if metrics.get('total_return',0)>=0.16:
                print(f'[arena] ALERT: {name} - {metrics["total_return"]:.2%} in 7d!',flush=True)
        except Exception as e:
            print(f'[arena] SKIP {name}: {e}',flush=True)
    time.sleep(180)
