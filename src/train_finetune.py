import argparse, json, math
import pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils.config import load_config, ensure_dir
from src.utils.seed import seed_everything
from src.data.dataset import ShardedEndoscopyDataset, load_labels, make_weighted_sampler
from src.models.siglip_classifier import SigLIPClassifier
from src.utils.metrics import compute_metrics, save_reports

def move_batch(batch, device): return {k:v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}

def evaluate(model, loader, labels, device, precision='bf16'):
    model.eval(); ys=[]; ps=[]; probs=[]; dtype=torch.bfloat16 if precision=='bf16' else torch.float16
    with torch.no_grad():
        for batch in tqdm(loader, desc='eval', leave=False):
            batch=move_batch(batch, device)
            with torch.autocast('cuda', dtype=dtype, enabled=device.type=='cuda'):
                logits=model(batch['pixel_values'])['logits']
            pr=torch.softmax(logits.float(), dim=-1); ys.extend(batch['labels'].cpu().tolist()); ps.extend(pr.argmax(-1).cpu().tolist()); probs.append(pr.cpu())
    return compute_metrics(ys,ps,labels), ys, ps, torch.cat(probs).numpy()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config', required=True); a=p.parse_args(); cfg=load_config(a.config); seed_everything(int(cfg.get('seed',42)))
    out_dir=ensure_dir(cfg['paths']['output_dir']); labels,label_to_id=load_labels(cfg['paths']['label_list']); use_h=cfg['model'].get('use_hierarchy',False)
    train_ds=ShardedEndoscopyDataset(cfg['paths']['split_csv'], cfg['paths']['data_root'], label_to_id, 'train', True, cfg['model']['image_size'], use_h)
    val_ds=ShardedEndoscopyDataset(cfg['paths']['split_csv'], cfg['paths']['data_root'], label_to_id, 'val', False, cfg['model']['image_size'], use_h)
    sampler=make_weighted_sampler(train_ds.df,label_to_id,cfg['training'].get('class_weighting','inverse_sqrt')) if cfg['training'].get('weighted_sampler',True) else None
    nw=cfg['training']['num_workers']; train_loader=DataLoader(train_ds,batch_size=cfg['training']['batch_size'],sampler=sampler,shuffle=(sampler is None),num_workers=nw,pin_memory=True,persistent_workers=nw>0)
    val_loader=DataLoader(val_ds,batch_size=cfg['training']['batch_size']*2,shuffle=False,num_workers=nw,pin_memory=True,persistent_workers=nw>0)
    model=SigLIPClassifier(cfg['model']['name'],len(labels),cfg['model'].get('freeze_encoder',False),use_h,len(train_ds.organ_to_id),len(train_ds.category_to_id),len(train_ds.family_to_id))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device)
    counts=train_ds.df['label'].map(label_to_id).value_counts().sort_index(); w=torch.tensor([1.0/(counts.get(i,1)**0.5) for i in range(len(labels))],dtype=torch.float); w=(w/w.mean()).to(device)
    ce=nn.CrossEntropyLoss(weight=w,label_smoothing=float(cfg['training'].get('label_smoothing',0.0))); aux_ce=nn.CrossEntropyLoss(label_smoothing=0.02)
    enc=[]; head=[]
    for n,pv in model.named_parameters():
        if pv.requires_grad: (enc if n.startswith('backbone') else head).append(pv)
    opt=torch.optim.AdamW([{'params':enc,'lr':float(cfg['training']['lr_encoder'])},{'params':head,'lr':float(cfg['training']['lr_head'])}], weight_decay=float(cfg['training']['weight_decay']))
    total_steps=max(1, math.ceil(len(train_loader)/cfg['training']['grad_accum_steps'])*cfg['training']['epochs']); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=total_steps)
    best={'macro_f1':-1,'mcc':-1}; patience=0; hist=[]; dtype=torch.bfloat16 if cfg['training'].get('precision','bf16')=='bf16' else torch.float16
    for epoch in range(1,cfg['training']['epochs']+1):
        model.train(); opt.zero_grad(set_to_none=True); loss_sum=0
        for step,batch in enumerate(tqdm(train_loader,desc=f'epoch {epoch}'),1):
            batch=move_batch(batch,device)
            with torch.autocast('cuda', dtype=dtype, enabled=device.type=='cuda'):
                out=model(batch['pixel_values']); loss=ce(out['logits'],batch['labels'])
                if use_h:
                    lw=cfg.get('loss_weights',{}); loss=loss*float(lw.get('label',1.0))+float(lw.get('organ_region',0.3))*aux_ce(out['organ_logits'],batch['organ_labels'])+float(lw.get('category',0.3))*aux_ce(out['category_logits'],batch['category_labels'])+float(lw.get('confusing_family',0.2))*aux_ce(out['family_logits'],batch['family_labels'])
                loss=loss/cfg['training']['grad_accum_steps']
            loss.backward(); loss_sum += loss.item()*cfg['training']['grad_accum_steps']
            if step % cfg['training']['grad_accum_steps']==0:
                nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        metrics,y_true,y_pred,probs=evaluate(model,val_loader,labels,device,cfg['training'].get('precision','bf16'))
        row={'epoch':epoch,'train_loss':loss_sum/max(1,len(train_loader)),**metrics}; hist.append(row); pd.DataFrame(hist).to_csv(out_dir/'history.csv',index=False); print('VAL',row)
        improved=(metrics['macro_f1']>best['macro_f1']) or (metrics['macro_f1']==best['macro_f1'] and metrics['mcc']>best['mcc'])
        if improved:
            best=metrics; patience=0; torch.save({'model':model.state_dict(),'cfg':cfg,'labels':labels,'metrics':metrics,'epoch':epoch},out_dir/'best.pt'); save_reports(y_true,y_pred,probs,labels,out_dir,'val_best'); print('saved best',best)
        else:
            patience+=1
            if patience>=cfg['training'].get('early_stop_patience',999): print('early stopping'); break
    (out_dir/'best_metrics.json').write_text(json.dumps(best,indent=2)); print('DONE best:',best)
if __name__=='__main__': main()
