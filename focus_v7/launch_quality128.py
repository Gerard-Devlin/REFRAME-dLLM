"""Explicit user-authorized128HumanEval test, only GPU1 and fixed v7."""
import fcntl
import json
import os
import shutil
import subprocess
import time

from .launch_match import REPO,ROOT,PYTHON,UUID,DATA,digest,sources,write,guard


def main():
    with (ROOT/'gpu1_research.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=json.loads((ROOT/'focus_v7_free_generation_queue.json').read_text())
        assert previous['status']=='complete' and previous['exit_code']==0
        before=guard();frozen=sources()
        from .quality128 import load_baseline
        from focus_dllm.llada_common import MODEL_ID,REVISION
        from focus_dllm.tuning.competitors import select_samples
        baseline=load_baseline(ROOT,DATA[0],select_samples(DATA[0],128,0),MODEL_ID,REVISION)
        stamp=time.strftime('%Y%m%d_%H%M%S');output=ROOT/f'focus_v7_quality128_{stamp}'
        output.mkdir(exist_ok=False)
        snapshot=ROOT/'source_snapshots'/f'before_focus_v7_quality128_{stamp}';snapshot.mkdir(exist_ok=False)
        for name in frozen:
            destination=snapshot/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,destination)
        write(output/'launch_manifest.json',dict(sources=frozen,guard=before,snapshot=str(snapshot),
            dataset_sha256=digest(DATA[0]),baseline_sha256=baseline['frozen_sha256'],
            authorization='Human explicitly requested128questions versus existing FOCUS after six-dev outcome',
            scope='HumanEval256, same128development IDs, fixed v7, historicalFOCUS reused'))
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
        assert code==0,'CPU tests failed; no GPU started'
        guard();assert sources()==frozen
        queue=ROOT/'focus_v7_quality128_queue.json'
        command=[PYTHON,'-u','-m','focus_v7.quality128','--root',str(ROOT),'--third-party',
            str(ROOT/'third_party/pinned_20261001'),'--dataset',str(DATA[0]),'--output',str(output/'evaluation')]
        with (output/'evaluation.log').open('x') as log:
            child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(queue,dict(status='running',pid=child.pid,gpu=1,uuid=UUID,output=str(output),sources=frozen))
            code=child.wait()
        (output/'exit_code').write_text(f'{code}\n');assert sources()==frozen
        complete=code==0 and (output/'evaluation/complete').exists()
        summary=json.loads((output/'evaluation/summary.json').read_text()) if complete else None
        write(queue,dict(status='complete' if complete else 'failed',pid=child.pid,exit_code=code,gpu=1,
                        output=str(output),summary=summary,automatic_expansion=False))
        if not complete:raise RuntimeError('Failure preserved; no automatic retry')
        guard();(output/'complete').write_text('OK\n')


if __name__=='__main__':main()
