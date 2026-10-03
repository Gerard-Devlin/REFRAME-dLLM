"""PACT resource contract, including the user's explicit shared-GPU1 authorization."""
import json
from pathlib import Path
import subprocess
from focus_dllm.tuning.firebreak.launch import guard as exclusive_guard, digest, REPO, UUID


def guard(root, previous, *, allow_shared_gpu1=False):
    if not allow_shared_gpu1:
        return exclusive_guard(root,previous)
    # Preserve historical terminal/source/main checks. Only GPU exclusivity changes.
    old=root/'focus_v4_quality128_20261003'
    manifest=json.loads((old/'launch_manifest.json').read_text())
    code=int((root/'focus_v4_quality128_exit_code').read_text().strip())
    queue=json.loads((root/'focus_v4_quality128_queue.json').read_text())
    assert code in (0,1) and (not queue.get('pid') or not Path(f'/proc/{queue["pid"]}').exists())
    assert {p.name:digest(p) for p in (REPO/'focus_dllm/tuning').glob('*.py')} == manifest['source']
    assert all(digest(Path(p))==h for p,h in manifest['protected'].items())
    if code==1:
        assert queue['phase']=='gsm8k_256_flash' and previous['exit_code']==1 and previous['records_retained']==94
        assert 'IndexError: The shape of the mask [26]' in (old/'gsm8k_256_flash.log').read_text(errors='replace')
        assert len(list((old/'gsm8k_256_flash/records').glob('*.json')))==94
    main=Path((REPO/'focus_dllm/dllm-eval/runs/latest_run.txt').read_text().strip())
    state=json.loads((main/'elastic/state.json').read_text())
    assert (main/'exit_code').read_text().strip()=='0'
    assert state['max_total_gpus']==5 and state['used_gpus']==0 and not state['workers']
    assert (REPO/'focus_dllm/README.md').stat().st_size==0
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
                                    '--format=csv,noheader,nounits'],text=True)
    rows=[r.split(',') for r in output.splitlines() if r.split(',')[0].strip()=='1']
    assert len(rows)==1 and rows[0][1].strip()==UUID
    row=rows[0]; free=int(row[3]); assert free>=24576,'GPU1 has insufficient free memory; no other process may be stopped'
    return dict(gpu=1,uuid=UUID,previous_exit_code=code,previous_pid_gone=True,
                main_max5=True,main_used0=True,protected_sources_match=True,
                gpu1_memory_mb=int(row[2]),gpu1_free_mb=free,utilization_percent=int(row[4]),
                shared_gpu1_authorized=True,processes_stopped=False,
                latency_context='Shared GPU1; timing is observational, not dedicated-card speed evidence')
