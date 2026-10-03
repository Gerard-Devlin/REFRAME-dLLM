"""Exclusive GPU1 launch after the old pipeline is terminal, never a retry."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

REPO = Path('/home/xuyouwen/REFRAME-dLLM')
PYTHON = '/home/xuyouwen/.conda/envs/fastdllm311/bin/python'
UUID = 'GPU-43d49500-e070-14e1-48b2-6a468fa01f8b'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)); temp.replace(path)


def sources():
    return {str(p.relative_to(REPO)):digest(p)
            for p in (REPO/'focus_dllm/tuning').rglob('*.py')}


def guard(root, previous):
    old = root/'focus_v4_quality128_20261003'
    manifest = json.loads((old/'launch_manifest.json').read_text())
    exit_code = int((root/'focus_v4_quality128_exit_code').read_text().strip())
    queue = json.loads((root/'focus_v4_quality128_queue.json').read_text())
    pid = queue.get('pid')
    assert exit_code in (0, 1) and (not pid or not Path(f'/proc/{pid}').exists())
    assert {p.name:digest(p) for p in (REPO/'focus_dllm/tuning').glob('*.py')} == manifest['source']
    assert all(digest(Path(p)) == h for p,h in manifest['protected'].items())
    if exit_code == 1:
        # Explicitly preserved known terminal failure, not silently relabeled0.
        assert queue['phase'] == 'gsm8k_256_flash'
        log = (old/'gsm8k_256_flash.log').read_text(errors='replace')
        assert 'IndexError: The shape of the mask [26]' in log
        assert len(list((old/'gsm8k_256_flash/records').glob('*.json'))) == 94
        assert previous['exit_code'] == 1 and previous['records_retained'] == 94
    main = Path((REPO/'focus_dllm/dllm-eval/runs/latest_run.txt').read_text().strip())
    state = json.loads((main/'elastic/state.json').read_text())
    assert (main/'exit_code').read_text().strip() == '0'
    assert state['max_total_gpus'] == 5 and state['used_gpus'] == 0 and not state['workers']
    assert (REPO/'focus_dllm/README.md').stat().st_size == 0
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used',
                                  '--format=csv,noheader,nounits'],text=True)
    row = [r.split(',') for r in gpu.splitlines() if r.split(',')[0].strip() == '1']
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name',
                                        '--format=csv,noheader'],text=True)
    assert len(row) == 1 and row[0][1].strip() == UUID and int(row[0][2]) < 128
    assert UUID not in processes, 'GPU1 occupied; no process may be killed'
    return dict(gpu=1,uuid=UUID,previous_exit_code=exit_code,previous_pid_gone=True,
                main_max5=True,main_used0=True,protected_sources_match=True,
                gpu1_memory_mb=int(row[0][2]))


def main():
    root = Path((REPO/'focus_dllm/dllm-eval/runs/latest_tuning.txt').read_text().strip())
    deployment = json.loads((root/'firebreak_deployment.json').read_text())
    output = Path(deployment['output'])
    with (root/'gpu1_research.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        checks = guard(root, deployment['previous_failure'])
        frozen = sources()
        assert frozen == deployment['sources']
        output.mkdir(exist_ok=False)
        env = os.environ.copy()
        for name in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy',
                     'RANK','WORLD_SIZE','LOCAL_RANK','MASTER_ADDR','MASTER_PORT'):
            env.pop(name,None)
        env.update(CUDA_VISIBLE_DEVICES=UUID,FOCUS_RESEARCH_GPU_UUID=UUID,
            HF_HOME='/home/xuyouwen/hf_home_local',HF_HUB_CACHE='/home/xuyouwen/hf_hub_local',
            HF_DATASETS_CACHE='/home/xuyouwen/hf_home_local/datasets',HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_EVALUATE_OFFLINE='1',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONUTF8='1',
            PYTHONPATH=f'{REPO}/focus_dllm/dllm-eval:{REPO}')
        write(output/'launch_manifest.json',dict(guard=checks,sources=frozen,
            previous_failure=deployment['previous_failure'],snapshot=deployment['snapshot'],
            plan='Private provenance/cost diagnostic; no new online commits or full128 expansion'))
        with (output/'cpu_tests.log').open('w') as log:
            test = subprocess.run([PYTHON,'-m','unittest','focus_dllm.tuning.firebreak.test_firebreak','-v'],
                                  cwd=REPO,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
        assert test.returncode == 0, 'CPU tests failed; no GPU experiment launched'
        guard(root, deployment['previous_failure']); assert sources() == frozen
        development = json.loads((root/'cpu_frozen_development128_20261003.json').read_text())
        command = [PYTHON,'-u','-m','focus_dllm.tuning.firebreak.probe',
            '--third-party',str(root/'third_party/pinned_20261001'),'--datasets',
            *(development['datasets'][task]['path'] for task in ('humaneval','mbpp','math')),
            '--output',str(output/'probe')]
        with (output/'probe.log').open('x') as log:
            child = subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(root/'firebreak_queue.json',dict(status='running',pid=child.pid,gpu=1,uuid=UUID,
                output=str(output),started=time.time(),sources=frozen))
            code = child.wait()
        assert sources() == frozen, 'Running research source changed'
        (output/'probe_exit_code').write_text(str(code)+'\n')
        (root/'firebreak_exit_code').write_text(str(code)+'\n')
        complete = code == 0 and (output/'probe/complete').exists()
        write(root/'firebreak_queue.json',dict(status='complete' if complete else 'failed',
            gpu=1,output=str(output),exit_code=code,goal_achieved=False,
            automatic_expansion=False,automatic_retry=False))
        if not complete:
            raise RuntimeError(f'Private FIREBREAK failed with exit{code}; retain all data, no retry')
        guard(root, deployment['previous_failure'])


if __name__ == '__main__':
    main()
