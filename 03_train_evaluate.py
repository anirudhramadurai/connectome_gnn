import pickle, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

Path("results").mkdir(exist_ok=True)

with open("data/graphs.pkl","rb") as f: graphs=pickle.load(f)
with open("data/roi_meta.pkl","rb") as f: roi_meta=pickle.load(f)
meta=pd.read_csv("data/metadata.csv")
labels=np.array([g["y"] for g in graphs]); N=len(graphs)
networks=np.array(roi_meta["networks"])
NETS=list(dict.fromkeys(roi_meta["networks"]))
print(f"Subjects: {N}  ASD={labels.sum()}  CTRL={(labels==0).sum()}")

def extract_features(graph):
    x=graph["x"]; feats=[]
    for i,na in enumerate(NETS):
        for j,nb in enumerate(NETS):
            if j<=i: continue
            feats.append(x[networks==na,0].mean()-x[networks==nb,0].mean())
    for net in NETS: feats.append(x[networks==net,1].mean())
    for net in NETS: feats.append(x[networks==net,2].mean())
    return np.array(feats)

print("Extracting features...")
X=np.stack([extract_features(g) for g in graphs])
nan_count = np.isnan(X).sum()
print(f"Feature matrix: {X.shape}  (NaNs: {nan_count} — will be imputed)")

skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
fold_results=[]; all_probs=np.zeros(N); node_imp=np.zeros((graphs[0]["x"].shape[0],5))

pipeline=Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("clf",     GradientBoostingClassifier(n_estimators=100,max_depth=3,learning_rate=0.05,random_state=42))
])

print("\n5-Fold Cross-Validation (Gradient Boosting on connectome features)\n")
for fold,(tr,te) in enumerate(skf.split(X,labels)):
    pipeline.fit(X[tr],labels[tr])
    probs=pipeline.predict_proba(X[te])[:,1]; preds=(probs>=0.5).astype(int)
    all_probs[te]=probs
    acc=accuracy_score(labels[te],preds); auc=roc_auc_score(labels[te],probs)
    cm=confusion_matrix(labels[te],preds,labels=[0,1])
    sens=cm[1,1]/max(cm[1].sum(),1); spec=cm[0,0]/max(cm[0].sum(),1)
    print(f"  Fold {fold+1}: Acc={acc:.3f}  AUC={auc:.3f}  Sens={sens:.3f}  Spec={spec:.3f}")
    fold_results.append({"fold":fold+1,"acc":acc,"auc":auc,"sens":sens,"spec":spec,"cm":cm,"te_idx":te})
    ta=np.stack([graphs[i]["x"] for i in te if labels[i]==1])
    tc=np.stack([graphs[i]["x"] for i in te if labels[i]==0])
    node_imp+=np.nan_to_num(np.abs(ta.mean(0)-tc.mean(0)))

print("\n=== Summary ===")
rows=[]
for m in ["acc","auc","sens","spec"]:
    vals=[r[m] for r in fold_results]; mu,sd=np.mean(vals),np.std(vals)
    lbl={"acc":"Accuracy","auc":"AUC-ROC","sens":"Sensitivity (ASD)","spec":"Specificity (CTRL)"}[m]
    print(f"  {lbl:25s}: {mu:.3f} +/- {sd:.3f}")
    rows.append({"Metric":lbl,"Mean":round(mu,3),"SD":round(sd,3)})

pd.DataFrame(rows).to_csv("results/metrics.csv",index=False)
results={"fold_results":fold_results,"all_probs":all_probs,"labels":labels,"node_imp":node_imp/5}
with open("results/cv_results.pkl","wb") as f: pickle.dump(results,f)
print("\nSaved results/")
