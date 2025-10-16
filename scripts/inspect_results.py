from pathlib import Path
import pickle
import numpy as np

root = Path('.')
files = [root / 'results' / 'baseline_results.pkl', root / 'results' / 'scs_id_results.pkl']

for p in files:
    print('\n===', p.name, '===')
    if not p.exists():
        print(' MISSING:', p)
        continue
    with open(p,'rb') as f:
        data = pickle.load(f)
    print(' keys:', sorted(list(data.keys())))
    # show label/pred sample info
    for key in ('labels','predictions'):
        if key in data:
            arr = np.array(data[key])
            print(f" {key}: shape={arr.shape}, dtype={arr.dtype}, unique={np.unique(arr)[:10]}{'...' if len(np.unique(arr))>10 else ''}")
            # counts
            vals, counts = np.unique(arr, return_counts=True)
            for v,c in zip(vals,counts):
                print(f"   value={v}: count={c}")
        else:
            print(' ', key, 'not found')
    # compute confusion counts using binary conversion
    if 'labels' in data and 'predictions' in data:
        y_true = np.array(data['labels'])
        y_pred = np.array(data['predictions'])
        # try to coerce to 1/0 using >0 rule as code assumes
        y_true_b = (y_true > 0).astype(int)
        y_pred_b = (y_pred > 0).astype(int)
        tn = int(((y_true_b==0) & (y_pred_b==0)).sum())
        fp = int(((y_true_b==0) & (y_pred_b==1)).sum())
        fn = int(((y_true_b==1) & (y_pred_b==0)).sum())
        tp = int(((y_true_b==1) & (y_pred_b==1)).sum())
        total = tn+fp+fn+tp
        print(' confusion counts (tn, fp, fn, tp):', tn, fp, fn, tp, ' total=', total)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else None
        print(' computed FPR (fp / (fp+tn)) =', fpr)
    # threshold optimization if present
    if 'threshold_optimization' in data:
        print(' threshold_optimization keys:', list(data['threshold_optimization'].keys()))
        to = data['threshold_optimization']
        for k in ('original_fpr','optimized_fpr','fpr_reduction_percentage','optimal_threshold','optimized_tpr'):
            if k in to:
                print(f"  {k}: {to[k]}")

print('\nDone')