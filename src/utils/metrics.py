from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, confusion_matrix, accuracy_score

def compute_metrics(y_true,y_pred,labels,prefix=''):
    mp,mr,mf,_=precision_recall_fscore_support(y_true,y_pred,average='macro',zero_division=0); ip,ir,if1,_=precision_recall_fscore_support(y_true,y_pred,average='micro',zero_division=0)
    return {prefix+'accuracy':accuracy_score(y_true,y_pred), prefix+'macro_precision':mp, prefix+'macro_recall':mr, prefix+'macro_f1':mf, prefix+'micro_precision':ip, prefix+'micro_recall':ir, prefix+'micro_f1':if1, prefix+'mcc':matthews_corrcoef(y_true,y_pred)}

def save_reports(y_true,y_pred,probs,labels,out_dir,stem='test'):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True); metrics=compute_metrics(y_true,y_pred,labels)
    pd.DataFrame([metrics]).to_csv(out_dir/f'{stem}_metrics.csv', index=False)
    p,r,f,s=precision_recall_fscore_support(y_true,y_pred,labels=list(range(len(labels))),zero_division=0)
    pd.DataFrame({'label':labels,'precision':p,'recall':r,'f1':f,'support':s}).to_csv(out_dir/f'{stem}_per_class_metrics.csv', index=False)
    cm=confusion_matrix(y_true,y_pred,labels=list(range(len(labels)))); pd.DataFrame(cm,index=labels,columns=labels).to_csv(out_dir/f'{stem}_confusion_matrix.csv')
    pred=pd.DataFrame({'y_true':[labels[i] for i in y_true],'y_pred':[labels[i] for i in y_pred],'y_true_id':y_true,'y_pred_id':y_pred})
    if probs is not None: pred['confidence']=np.max(probs,axis=1)
    pred.to_csv(out_dir/f'{stem}_predictions.csv', index=False); return metrics
