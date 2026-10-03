"""Exclusive GPU1 launch, frozen PACT plus all reused research sources."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from .evaluate import implementation,write
from focus_dllm.tuning.firebreak.launch import guard,REPO,PYTHON,UUID


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',choices=('mechanism_smoke','development128'),required=True)
    args=parser.parse_args()
    root=Path((REPO/'focus_dllm/dllm-eval/runs/latest_tuning.txt').read_text().strip())
    deployment=json.loads((root/'pact_deployment.json').read_text())
    output=Path(deployment['output'])
    with (root/'gpu1_research.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=json.loads((root/'firebreak_base_audit_queue.json').read_text())
        assert previous['status']=='complete' and previous['exit_code']==0
        assert (Path(previous['output'])/'probe/complete').exists()
        checks=guard(root,deployment['previous_failure'])
        frozen=implementation();assert frozen==deployment['sources']
        assert args.scope==deployment['scope'],'No automatic scope expansion'
        output.mkdir(exist_ok=False)
        env=os.environ.copy()
        for name in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy',
                     'RANK','WORLD_SIZE','LOCAL_RANK','MASTER_ADDR','MASTER_PORT'):
            env.pop(name,None)
        env.update(CUDA_VISIBLE_DEVICES=UUID,FOCUS_RESEARCH_GPU_UUID=UUID,
            HF_HOME='/home/xuyouwen/hf_home_local',HF_HUB_CACHE='/home/xuyouwen/hf_hub_local',
            HF_DATASETS_CACHE='/home/xuyouwen/hf_home_local/datasets',HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_EVALUATE_OFFLINE='1',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONUTF8='1',
            PYTHONPATH=f'{REPO}/focus_dllm/dllm-eval:{REPO}')
        write(output/'launch_manifest.json',dict(guard=checks,sources=frozen,snapshot=deployment['snapshot'],
                                               scope=args.scope,automatic_expansion=False))
        with (output/'cpu_tests.log').open('x') as log:
            test=subprocess.run([PYTHON,'-m','unittest','pact_dllm.test_pact','-v'],cwd=REPO,
                env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
        assert test.returncode==0,'CPU checks failed; no model execution'
        guard(root,deployment['previous_failure']);assert implementation()==frozen
        development=json.loads((root/'cpu_frozen_development128_20261003.json').read_text())
        command=[PYTHON,'-u','-m','pact_dllm.evaluate','--third-party',str(root/'third_party/pinned_20261001'),
            '--datasets',*(development['datasets'][t]['path'] for t in ('humaneval','mbpp','math')),
            '--output',str(output/'evaluation'),'--scope',args.scope,'--limit',
            '2' if args.scope=='mechanism_smoke' else '128','--length','64' if args.scope=='mechanism_smoke' else '256']
        with (output/'evaluation.log').open('x') as log:
            child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(root/'pact_queue.json',dict(status='running',pid=child.pid,gpu=1,uuid=UUID,
                output=str(output),scope=args.scope,started=time.time(),sources=frozen))
            code=child.wait()
        unchanged=implementation()==frozen
        (output/'exit_code').write_text(str(code)+'\n');(root/'pact_exit_code').write_text(str(code)+'\n')
        complete=code==0 and unchanged and (output/'evaluation/complete').exists()
        write(root/'pact_queue.json',dict(status='complete' if complete else 'failed',output=str(output),
            scope=args.scope,gpu=1,exit_code=code,sources_unchanged=unchanged,
            goal_achieved=False,automatic_expansion=False,automatic_retry=False))
        if not complete:raise RuntimeError(f'PACT exit{code}; records preserved, no automatic retry')
        guard(root,deployment['previous_failure'])


if __name__=='__main__':main()
