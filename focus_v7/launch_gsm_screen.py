"""Run one unchanged coverage-age128 GSM8K screen on physicalGPU1."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

from .launch_match import REPO, ROOT, PYTHON, UUID, sources, write, guard


def main():
    with (ROOT/'gpu1_research.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads((ROOT/'focus_v7_handoff_queue.json').read_text())
        old = Path(previous['output'])
        assert previous['status'] == 'complete' and previous['exit_code'] == 0
        assert not Path('/proc/'+str(previous['pid'])).exists() and (old/'complete').exists()
        queue = ROOT/'focus_v7_gsm128_queue.json'
        assert not queue.exists(), 'Never overwrite or repeat this screen'
        before = guard()
        frozen = sources()
        stamp = time.strftime('%Y%m%d_%H%M%S')
        output = ROOT/f'focus_v7_gsm128_{stamp}'
        output.mkdir(exist_ok=False)
        snapshot = ROOT/'source_snapshots'/f'before_focus_v7_gsm128_{stamp}'
        snapshot.mkdir(exist_ok=False)
        for name in frozen:
            dest = snapshot/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO/name, dest)
        with tarfile.open(output/'tested_source.tar', 'w') as archive:
            for name in frozen:
                archive.add(REPO/name, arcname=name)
        data = json.loads((ROOT/'cpu_frozen_development128_20261003.json').read_text())['datasets']['gsm8k']['path']
        write(output/'launch_manifest.json', dict(sources=frozen, guard=before, snapshot=str(snapshot),
            dataset=data, parent_terminal=str(old), scope='One fixed GSM8K128 screen; no baseline regeneration'))
        env = os.environ.copy()
        for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy',
                    'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT'):
            env.pop(key, None)
        env.update(CUDA_VISIBLE_DEVICES=UUID, FOCUS_RESEARCH_GPU_UUID=UUID,
            HF_HOME='/home/xuyouwen/hf_home_local', HF_HUB_CACHE='/home/xuyouwen/hf_hub_local',
            HF_DATASETS_CACHE='/home/xuyouwen/hf_home_local/datasets', HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', HF_EVALUATE_OFFLINE='1',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', PYTHONUTF8='1',
            PYTHONPATH=f'{REPO}/focus_dllm/dllm-eval:{REPO}')
        with (output/'cpu_tests.log').open('x') as log:
            code = subprocess.run([PYTHON, '-m', 'unittest', 'discover', '-s', 'focus_v7/tests', '-v'],
                cwd=REPO, env=dict(env, CUDA_VISIBLE_DEVICES=''), stdout=log, stderr=subprocess.STDOUT).returncode
        assert code == 0, 'CPU checks failed; no GPU launched'
        guard()
        assert sources() == frozen
        command = [PYTHON, '-u', '-m', 'focus_v7.gsm_screen', '--third-party',
            str(ROOT/'third_party/pinned_20261001'), '--dataset', data, '--root', str(ROOT),
            '--output', str(output/'evaluation')]
        with (output/'evaluation.log').open('x') as log:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            write(queue, dict(status='running', pid=child.pid, output=str(output), gpu=1, uuid=UUID, sources=frozen))
            code = child.wait()
        (output/'exit_code').write_text(f'{code}\n')
        assert sources() == frozen, 'Research source changed during generation'
        complete = code == 0 and (output/'evaluation/complete').exists()
        summary = json.loads((output/'evaluation/summary.json').read_text()) if complete else None
        write(queue, dict(status='complete' if complete else 'failed', pid=child.pid, exit_code=code,
            output=str(output), gpu=1, uuid=UUID, summary=summary, automatic_expansion=False))
        if not complete:
            raise RuntimeError('Failed screen retained; no automatic restart')
        guard()
        (output/'complete').write_text('OK\n')


if __name__ == '__main__':
    main()
