import sys, json, os
sys.path.insert(0, '.')
from gpu_conv_trainer import solve_task_gpu
from grammar.primitives import score_model, verify_model
from pathlib import Path
ROOT = Path('.').resolve().parent
start = int(os.environ.get('START', '1'))
end = int(os.environ.get('END', '80'))
for tn in range(start, end+1):
    tf = ROOT / f'task{tn:03d}.json'
    if not tf.exists(): continue
    with open(tf) as f:
        task = json.load(f)
    try:
        r = solve_task_gpu(task, tn, 'cuda', num_seeds=8, max_time_s=180)
        if r.get('status') == 'solved':
            model = r['model']
            c, t = verify_model(model, task)
            if c == t and t > 0:
                s = score_model(model)
                print(json.dumps({'task': tn, 'status': 'solved', 'cost': s['cost'], 'arch': r.get('arch','conv')}), flush=True)
            else:
                print(json.dumps({'task': tn, 'status': 'verify_fail'}), flush=True)
        else:
            print(json.dumps({'task': tn, 'status': 'unsolved'}), flush=True)
    except Exception as e:
        print(json.dumps({'task': tn, 'status': 'error', 'msg': str(e)[:100]}), flush=True)
