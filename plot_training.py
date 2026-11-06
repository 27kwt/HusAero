"""Export publication-ready per-epoch metrics/loss plots from saved CSV evidence."""
import argparse,csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--models',nargs='+',default=['HusAero-8n','HusAero-11n']);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);series={};metadata=[]
    for model in a.models:
        file=a.run/model/'results.csv'
        if not file.exists():continue
        rows=list(csv.DictReader(file.open()));series[model]=rows
        metadata.append(dict(model=model,csv=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest(),epochs=len(rows)))
        fig,axes=plt.subplots(2,5,figsize=(17,6.8),layout='constrained')
        fields=['train/box_loss','train/cls_loss','train/dfl_loss','metrics/precision(B)','metrics/recall(B)',
                'val/box_loss','val/cls_loss','val/dfl_loss','metrics/mAP50(B)','metrics/mAP50-95(B)']
        for ax,key in zip(axes.flat,fields):
            ax.plot([int(r['epoch']) for r in rows],[float(r[key]) for r in rows],color='#167b6a',linewidth=1.6)
            ax.set_title(key.replace('metrics/','').replace('(B)',''));ax.set_xlabel('Epoch');ax.grid(alpha=.2)
            if key.startswith('metrics/'):ax.set_ylim(0,1)
        fig.suptitle(model+' — training and validation history (test results are separate)')
        fig.savefig(a.out/(model+'_epochs.png'),dpi=180);fig.savefig(a.out/(model+'_epochs.pdf'));plt.close(fig)
        with plt.rc_context({'font.size':14}):
            fig,axes=plt.subplots(2,2,figsize=(10,7.4),layout='constrained')
            for ax,key in zip(axes.flat,['metrics/precision(B)','metrics/recall(B)','metrics/mAP50(B)','metrics/mAP50-95(B)']):
                ax.plot([int(r['epoch']) for r in rows],[float(r[key]) for r in rows],color='#167b6a',linewidth=1.8)
                ax.set_title(key.replace('metrics/','').replace('(B)',''));ax.set_xlabel('Epoch');ax.set_ylim(0,1);ax.grid(alpha=.2)
            fig.suptitle(model+' validation metrics');fig.savefig(a.out/(model+'_metrics.png'),dpi=200);plt.close(fig)
            fig,axes=plt.subplots(3,2,figsize=(10,10),layout='constrained')
            for row,key in enumerate(['box_loss','cls_loss','dfl_loss']):
                for col,split in enumerate(['train','val']):
                    ax=axes[row,col];ax.plot([int(r['epoch']) for r in rows],[float(r[split+'/'+key]) for r in rows],color='#2467bf',linewidth=1.8)
                    ax.set_title(split+'/'+key);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
            fig.suptitle(model+' losses');fig.savefig(a.out/(model+'_losses.png'),dpi=200);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,key in zip(axes.flat,['metrics/precision(B)','metrics/recall(B)','metrics/mAP50(B)','metrics/mAP50-95(B)']):
        for (name,rows),color in zip(series.items(),['#167b6a','#2467bf']):
            ax.plot([int(r['epoch']) for r in rows],[float(r[key]) for r in rows],label=name,color=color)
        ax.set_title(key.replace('metrics/','').replace('(B)',''));ax.set_xlabel('Epoch');ax.set_ylim(0,1);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Validation curves - check starting-history differences')
    fig.savefig(a.out/'model_comparison_curves.png',dpi=180);fig.savefig(a.out/'model_comparison_curves.pdf');plt.close(fig)
    (a.out/'plot_evidence.json').write_text(json.dumps(metadata,indent=2));print(a.out)
if __name__=='__main__':main()
