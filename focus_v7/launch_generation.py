"""Guarded six-dev task-scored diagnosis; no automatic expansion/retry."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .launch_match import REPO,ROOT,PYTHON,UUID,DATA,REFERENCE,digest,sources,write,guard


def main():
    with (ROOT/'gpu1_research.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=json.loads((ROOT/'focus_v7_match_cache_queue.json').read_text())
        assert previous['status']=='complete' and previous['exit_code']==0
        # Preserve failed local agreement gate. This distinct tiny task-scored
        # diagnosis is within the user's explicit allowance for changed answers.
        assert previous['gate_passed'] is False
        before=guard(); frozen=sources()
        stamp=time.strftime('%Y%m%d_%H%M%S')
        output=ROOT/f'focus_v7_free_generation_{stamp}'
        output.mkdir(exist_ok=False)
        snapshot=ROOT/'source_snapshots'/f'before_focus_v7_free_generation_{stamp}'
        snapshot.mkdir(exist_ok=False)
        for name in frozen:
            destination=snapshot/name; destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(REPO/name,destination)
        write(output/'launch_manifest.json',dict(sources=frozen,guard=before,snapshot=str(snapshot),
            datasets={str(p):digest(p) for p in DATA},reference=digest(REFERENCE),
            protocol=digest(REPO/'focus_v7/GENERATION_PLAN.md'),
            predecessor_gate='failed token-reference gate unchanged; objective scoring diagnosis',
            scope='six reused dev prompts, official task score plus three fixed-work pairs'))
        env=os.environ.copy()
        for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy',
                    'RANK','WORLD_SIZE','LOCAL_RANK','MASTER_ADDR','MASTER_PORT'):
            env.pop(key,None)
        env.update(CUDA_VISIBLE_DEVICES=UUID,FOCUS_RESEARCH_GPU_UUID=UUID,
            HF_HOME='/home/xuyouwen/hf_home_local',HF_HUB_CACHE='/home/xuyouwen/hf_hub_local',
            HF_DATASETS_CACHE='/home/xuyouwen/hf_home_local/datasets',HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_EVALUATE_OFFLINE='1',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONUTF8='1',
            PYTHONPATH=f'{REPO}/focus_dllm/dllm-eval:{REPO}')
        with (output/'cpu_tests.log').open('x') as log:
            tests=subprocess.run([PYTHON,'-m','unittest','discover','-s','focus_v7/tests','-v'],
                cwd=REPO,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
        assert tests.returncode==0,'CPU test failed; no GPU task launched'
        guard(); assert sources()==frozen
        queue=ROOT/'focus_v7_free_generation_queue.json'
        command=[PYTHON,'-u','-m','focus_v7.free_generation','--third-party',str(ROOT/'third_party/pinned_20261001'),
            '--datasets',*map(str,DATA),'--reference',str(REFERENCE),'--output',str(output/'generation')]
        with (output/'generation.log').open('x') as log:
            child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(queue,dict(status='running',pid=child.pid,gpu=1,uuid=UUID,output=str(output),sources=frozen))
            code=child.wait()
        (output/'exit_code').write_text(f'{code}\n')
        assert sources()==frozen,'research source changed during live GPU generation'
        complete=code==0 and (output/'generation/complete').exists()
        report=json.loads((output/'generation/summary.json').read_text()) if complete else {}
        write(queue,dict(status='complete' if complete else 'failed',pid=child.pid,exit_code=code,
            output=str(output),gpu=1,summary=report.get('task_summary'),automatic_expansion=False))
        if not complete:
            raise RuntimeError('failure preserved; no automatic retry')
        guard(); (output/'complete').write_text('OK\n')


if __name__=='__main__':
    main()
