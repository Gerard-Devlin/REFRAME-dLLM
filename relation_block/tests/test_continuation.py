from argparse import Namespace
from pathlib import Path
import pytest
from relation_block.continuation import milestones, training_command
from relation_block.full_state import epoch_batches


def test_milestones_same_seed_no_replacement_and_batch_overshoot():
    rows = [dict(ids=list(range(11 + i))) for i in range(20)]
    batches = epoch_batches(len(rows), 6, 1234)
    points = milestones(rows, batches, 20, (50, 150, 250))
    assert len(set(i for b in batches for i in b)) == len(rows)
    for p in points:
        consumed = sum(min(len(rows[i]['ids']), 20) for b in batches[:p['step']] for i in b)
        before = sum(min(len(rows[i]['ids']), 20) for b in batches[:p['step']-1] for i in b)
        assert before < p['requested_tokens'] <= consumed == p['original_tokens']
    with pytest.raises(ValueError, match='smaller'):
        milestones(rows, batches, 20, (10000,))


def test_pause_resume_preserves_full_epoch_schedule():
    args = Namespace(world_size=6, data=Path('data'), lr=2e-5, global_batch=12, seed=1234)
    a = training_command(args, Path('train'), 40)
    b = training_command(args, Path('train'), 80, Path('train/step_00000040'))
    for cmd in (a, b):
        assert cmd[cmd.index('--steps')+1] == '0'
        assert cmd[cmd.index('--eval-every')+1] == '0'
        assert '--smoke' not in cmd
    assert '--resume' not in a
    assert b[b.index('--resume')+1] == str(Path('train/step_00000040'))


def test_campaign_runs_both_arms_and_every_observation(tmp_path, monkeypatch):
    import json
    from relation_block import continuation as c
    data, output = tmp_path / 'data', tmp_path / 'campaign'
    data.mkdir()
    (data / 'train.json').write_text(json.dumps([dict(ids=[1, 2])] * 6))
    (data / 'full_smoke.json').write_text(json.dumps(dict(
        pass_=True, implementation={}, data_hash='hash', world_size=6,
        global_batch=12, micro_batch=1)).replace('"pass_"', '"pass"'))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5')
    monkeypatch.setattr(c, 'require_gate', lambda _: None)
    monkeypatch.setattr(c, 'manifest', lambda _: dict(length=2))
    monkeypatch.setattr(c, 'digest', lambda _: 'hash')
    monkeypatch.setattr(c, 'implementation_hashes', lambda: {})
    points = [dict(requested_tokens=x, original_tokens=x+1, step=i)
              for i, x in enumerate((500000, 1000000, 2000000), 1)]
    monkeypatch.setattr(c, 'milestones', lambda *a: points)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        dest = Path(cmd[cmd.index('--output')+1])
        if 'relation_block.train' in cmd:
            step = int(cmd[cmd.index('--stop-after')+1])
            checkpoint = dest / f'step_{step:08d}'
            checkpoint.mkdir(parents=True)
            (checkpoint / 'metadata.json').write_text(json.dumps(dict(
                original_tokens=points[step-1]['original_tokens'], completed_steps=step)))
            (dest / 'latest.json').write_text(json.dumps(dict(checkpoint=checkpoint.name)))
        elif 'relation_block.evaluate' in cmd:
            dest.mkdir(parents=True)
            (dest / 'summary.json').write_text('{"results": {}}')
        else:
            dest.write_text('{}')

    monkeypatch.setattr(c.subprocess, 'run', fake_run)
    args = Namespace(data=data, output=output, world_size=6, global_batch=12,
                     seed=1234, limit=256, rounds='8,16', reconstruction_limit=32)
    c.campaign(args)
    training = [cmd for cmd in calls if 'relation_block.train' in cmd]
    assert len(training) == 6
    assert [('--resume' in cmd) for cmd in training] == [False, True, True] * 2
    assert [cmd[cmd.index('--lr')+1] for cmd in training] == ['2e-05'] * 3 + ['4e-06'] * 3
    result = json.loads((output / 'summary.json').read_text())
    assert result['complete'] and len(result['results']) == 7
