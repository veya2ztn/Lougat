import os
import wandb
import sys
from tbparse import SummaryReader
from project_arguements import get_args
from train_via_accelerate import build_accelerator
import wandb,sys,os
trial_path = sys.argv[1]
ROOTDIR    = os.path.dirname(trial_path.rstrip('/'))
args = get_args(os.path.join(ROOTDIR,'train_config.json'),args=['--task','train','--model','flougatU_small','--Dataset', 'uparxive','--use_wandb'])
args.output_dir = ROOTDIR
accelerator = build_accelerator(args)
summary_dir = trial_path
outdf = []
for filename in os.listdir(summary_dir):
    if 'event' not in filename:continue
    log_dir = os.path.join(summary_dir,filename)
    print(log_dir)
    reader = SummaryReader(log_dir)
    df = reader.scalars

    if len(df) < 1: 
        print(f"==>no scalars at,pass")
        continue
    print("===>start parsing tensorboard..............")
    outdf.append(df)
assert len(outdf)>=1

step = 0
all_pool={}
for df in outdf:
    for _, tag, val in df.values:
        if step not in all_pool:
            all_pool[step]={}
        else:
            print(f"repeat at step {step} tag {tag} val {val}")
        all_pool[step][tag]  =val 
        step+=1

for step, record in all_pool.items():
    record['step'] = step
    wandb.log(record,step =step)
