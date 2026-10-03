"""Guarded physicalGPU1 query-budget screen; no automatic expansion."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .launch_match import REPO,ROOT,PYTHON,UUID,DATA,sources,write,guard


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repair-attempt',choices=('compact-readout',),default=None)
    args=parser.parse_args()
    with (ROOT/'gpu1_research.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=json.loads((ROOT/'focus_v7_revision16_queue.json').read_text())
        old=Path(previous['output'])
        assert previous['status']=='complete' and previous['exit_code']==0 and (old/'complete').exists()
        assert not Path('/proc/'+str(previous['pid'])).exists()
        if args.repair_attempt:
            failed=json.loads((ROOT/'focus_v7_budget16_queue.json').read_text())
            assert failed['status']=='failed' and failed['exit_code']==1
            assert not Path('/proc/'+str(failed['pid'])).exists()
        before=guard();frozen=sources()
        stamp=time.strftime('%Y%m%d_%H%M%S')
        suffix='readoutfix1_' if args.repair_attempt else ''
        output=ROOT/f'focus_v7_budget16_{suffix}{stamp}'
        output.mkdir(exist_ok=False)
        snapshot=ROOT/'source_snapshots'/f'before_focus_v7_budget16_{stamp}';snapshot.mkdir(exist_ok=False)
        for name in frozen:
            dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,dest)
        write(output/'launch_manifest.json',dict(sources=frozen,guard=before,snapshot=str(snapshot),
            old_revision=str(old),scope='Fixed16development clean-coverage/age screen; no baseline regeneration'))
        env=os.environ.copy()
        for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy',
                    'RANK','WORLD_SIZE','LOCAL_RANK','MASTER_ADDR','MASTER_PORT'):env.pop(key,None)
        env.update(CUDA_VISIBLE_DEVICES=UUID,FOCUS_RESEARCH_GPU_UUID=UUID,
            HF_HOME='/home/xuyouwen/hf_home_local',HF_HUB_CACHE='/home/xuyouwen/hf_hub_local',
            HF_DATASETS_CACHE='/home/xuyouwen/hf_home_local/datasets',HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_EVALUATE_OFFLINE='1',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONUTF8='1',
            PYTHONPATH=f'{REPO}/focus_dllm/dllm-eval:{REPO}')
        with (output/'cpu_tests.log').open('x') as log:
            code=subprocess.run([PYTHON,'-m','unittest','discover','-s','focus_v7/tests','-v'],
                cwd=REPO,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
        assert code==0,'CPU checks failed; no GPU launched'
        guard();assert sources()==frozen
        queue=ROOT/('focus_v7_budget16_readoutfix1_queue.json' if args.repair_attempt else 'focus_v7_budget16_queue.json')
        assert not queue.exists(),'do not duplicate or silently replace previous budget experiment'
        command=[PYTHON,'-u','-m','focus_v7.budget_evaluate','--third-party',str(ROOT/'third_party/pinned_20261001'),
            '--dataset',str(DATA[0]),'--root',str(ROOT),'--old-revision',str(old),'--output',str(output/'evaluation')]
        with (output/'evaluation.log').open('x') as log:
            child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(queue,dict(status='running',pid=child.pid,gpu=1,uuid=UUID,output=str(output),sources=frozen))
            code=child.wait()
        (output/'exit_code').write_text(f'{code}\n');assert sources()==frozen
        complete=code==0 and (output/'evaluation/complete').exists()
        summary=json.loads((output/'evaluation/summary.json').read_text()) if complete else None
        write(queue,dict(status='complete' if complete else 'failed',pid=child.pid,exit_code=code,gpu=1,
            output=str(output),summary=summary,automatic_expansion=False))
        if not complete:raise RuntimeError('Failed experiment retained; no automatic restart')
        guard();(output/'complete').write_text('OK\n')


if __name__=='__main__':main()
