import argparse, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils.config import load_config
from src.data.dataset import ShardedEndoscopyDataset, load_labels
from src.models.siglip_classifier import SigLIPClassifier
from src.utils.metrics import save_reports

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config', required=True); a=p.parse_args(); cfg=load_config(a.config)
    labels,label_to_id=load_labels(cfg['paths']['label_list']); use_h=cfg['model'].get('use_hierarchy',False)
    ds=ShardedEndoscopyDataset(cfg['paths']['split_csv'],cfg['paths']['data_root'],label_to_id,cfg['eval'].get('split','test'),False,cfg['model']['image_size'],use_h)
    loader=DataLoader(ds,batch_size=cfg['eval']['batch_size'],shuffle=False,num_workers=cfg['eval']['num_workers'],pin_memory=True)
    ckpt=torch.load(cfg['paths']['checkpoint'],map_location='cpu'); model=SigLIPClassifier(cfg['model']['name'],len(labels),False,use_h,len(ds.organ_to_id),len(ds.category_to_id),len(ds.family_to_id)); model.load_state_dict(ckpt['model'],strict=False)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device).eval(); ys=[]; ps=[]; probs=[]
    with torch.no_grad():
        for batch in tqdm(loader,desc='test'):
            x=batch['pixel_values'].to(device); y=batch['labels'].to(device); logits=model(x)['logits'].float()
            if cfg['eval'].get('tta_hflip',False): logits=(logits+model(torch.flip(x,dims=[3]))['logits'].float())/2
            pr=torch.softmax(logits,dim=-1); ys.extend(y.cpu().tolist()); ps.extend(pr.argmax(-1).cpu().tolist()); probs.append(pr.cpu())
    metrics=save_reports(ys,ps,torch.cat(probs).numpy(),labels,cfg['paths']['output_dir'],cfg['eval'].get('split','test')); print(metrics)
if __name__=='__main__': main()
