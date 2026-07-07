#!/usr/bin/env python3
"""CryptAI Clean Trainer v4 — full training with MCC, precision, recall, F1, confusion matrix"""
import os,sys,time,glob,json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, precision_recall_fscore_support, confusion_matrix

BASE="/mnt/hive_storage/CryptAI"
sys.path.insert(0,f"{BASE}/lib")
from features import build_features
os.makedirs(f"{BASE}/training/checkpoints",exist_ok=True)

model_type=sys.argv[1] if len(sys.argv)>1 else "transformer"
ARCHS={
    "transformer":{"bs":2048,"lr":1e-4,"args":[256,8,6,96], "class":"DeepTransformer"},
    "tcn":{"bs":8192,"lr":1e-3,"args":[512,8,96], "class":"TCN"},
    "lstm":{"bs":1024,"lr":1e-3,"args":[256,2,96], "class":"DeepLSTM"},
}
cfg=ARCHS.get(model_type,ARCHS["transformer"])
BATCH=cfg["bs"];LR=cfg["lr"]
print(f"[train] {model_type} | batch={BATCH} lr={LR} | {cfg['class']}",flush=True)

# Load data (temporal, z-score labels)
all_X,all_y,all_ts=[],[],[]
for f in sorted(glob.glob(f"{BASE}/data/*/*/historical.csv")):
    try:
        df=pd.read_csv(f,parse_dates=["timestamp"])
        if len(df)<200:continue
        feats=build_features(df)
        if len(feats)<196:continue
        close=df["close"].values[-len(feats):];vals=feats.values.astype(np.float32);ts=df["timestamp"].values[-len(feats):]
        ret=close[1:]/close[:-1]-1
        w=min(100,len(ret));mr=np.convolve(ret,np.ones(w)/w,mode="same")
        sr=np.sqrt(np.convolve(ret**2,np.ones(w)/w,mode="same")-mr**2)+1e-8
        z=(ret-mr)/sr;labels=np.where(z<-1.0,0,np.where(z>1.0,2,1))
        for i in range(len(vals)-96):
            all_X.append(vals[i:i+96]);all_y.append(int(labels[i+96]));all_ts.append(ts[i+96])
    except:pass

X=np.array(all_X);y=np.array(all_y);ts=np.array(all_ts);del all_X,all_y,all_ts
order=np.argsort(ts);X=X[order];y=y[order];ts=ts[order]
n=len(X);s=int(n*0.7);vs=int(n*0.85)
X_train,X_val,X_test=X[:s],X[s:vs],X[vs:];y_train,y_val,y_test=y[:s],y[s:vs],y[vs:]
mean=X_train.mean(axis=(0,1),keepdims=True);std=X_train.std(axis=(0,1),keepdims=True)+1e-8
X_train=(X_train-mean)/std;X_val=(X_val-mean)/std;X_test=(X_test-mean)/std
del X,order

# Class weights
cw=np.bincount(y_train);cw=1.0/(cw.astype(float)+1e-8);cw=cw/cw.sum()*len(cw)
print(f"  Train:{len(X_train)} Val:{len(X_val)} Test:{len(X_test)} | Weights:{cw.round(3).tolist()}",flush=True)

# Model
input_dim=X_train.shape[2]
if model_type=="transformer":
    from models_v2 import DeepTransformer as M;m=M(input_dim,*cfg["args"]).to("cuda")
elif model_type=="tcn":
    from models_v2 import TCN as M;m=M(input_dim,*cfg["args"]).to("cuda")
elif model_type=="lstm":
    from models_v2 import DeepLSTM as M;m=M(input_dim,*cfg["args"]).to("cuda")
else:
    from models import MLPDirection as M;m=M(input_dim*96).to("cuda")

print(f"  Params: {sum(p.numel() for p in m.parameters()):,}",flush=True)
opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-5)
sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=5,factor=0.5)
crit=torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(cw).to("cuda"))

def compute_metrics(y_true, y_pred):
    ba=balanced_accuracy_score(y_true,y_pred)
    mcc=matthews_corrcoef(y_true,y_pred)
    cm=confusion_matrix(y_true,y_pred,labels=[0,1,2])
    p,r,f1,_=precision_recall_fscore_support(y_true,y_pred,labels=[0,1,2],zero_division=0)
    return {"balanced_acc":round(ba,4),"mcc":round(mcc,4),
            "precision":{0:round(p[0],3),1:round(p[1],3),2:round(p[2],3)},
            "recall":{0:round(r[0],3),1:round(r[1],3),2:round(r[2],3)},
            "f1":{0:round(f1[0],3),1:round(f1[1],3),2:round(f1[2],3)},
            "confusion":cm.tolist()}

best_mcc,best_metrics,best_state=-1,None,None;pat,t0=0,time.time()
for ep in range(100):
    for i in range(0,len(X_train),BATCH):
        idx=slice(i,min(i+BATCH,len(X_train)))
        xb=torch.from_numpy(X_train[idx]).to("cuda");yb=torch.from_numpy(y_train[idx]).to("cuda").long()
        opt.zero_grad(set_to_none=True);crit(m(xb),yb).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
    
    m.eval();metrics={}
    for name,Xd,yd in [("train",X_train,y_train),("val",X_val,y_val),("test",X_test,y_test)]:
        all_p=[];vl_tot=nb=0
        with torch.no_grad():
            for j in range(0,len(Xd),2048):
                xv=torch.from_numpy(Xd[j:j+2048]).to("cuda")
                yv_=torch.from_numpy(yd[j:j+2048]).to("cuda").long()
                o=m(xv);vl_tot+=crit(o,yv_).item()*len(xv)
                all_p.extend(o.argmax(1).cpu().numpy());nb+=len(xv)
        met=compute_metrics(yd[:len(all_p)],np.array(all_p))
        met["loss"]=round(vl_tot/nb,4)
        metrics[name]=met
    m.train();sched.step(metrics["val"]["loss"])
    
    vu=metrics["val"];tm=metrics["test"]
    if vu["mcc"]>best_mcc:best_mcc=vu["mcc"];best_metrics=vu;best_state=m.state_dict();pat=0
    else:pat+=0.5
    vram=torch.cuda.memory_allocated(0)/1024/1024/1024
    print(f"{model_type.upper()} EP{ep:>3} | VBalAcc:{vu['balanced_acc']:.4f} VMCC:{vu['mcc']:.4f} | TeBalAcc:{tm['balanced_acc']:.4f} TeMCC:{tm['mcc']:.4f} | VRAM:{vram:.2f}GB | {time.time()-t0:.0f}s",flush=True)
    if pat>=15:print(f"Early stop EP{ep}",flush=True);break

if best_state:
    torch.save({"state":best_state,"val_metrics":best_metrics,"test_metrics":metrics["test"],"config":cfg},f"{BASE}/training/checkpoints/{model_type}_champion.pth")
    print(f"Saved {model_type}_champion.pth | Val MCC={best_mcc}",flush=True)
    print(f"Val: {json.dumps(best_metrics,indent=2)}",flush=True)
    print(f"Test: {json.dumps(metrics['test'],indent=2)}",flush=True)
print(f"Done: {model_type} | {time.time()-t0:.0f}s",flush=True)