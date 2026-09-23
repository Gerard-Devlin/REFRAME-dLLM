import json
import sys
from requests.exceptions import ChunkedEncodingError

from denoising_dag import download


def test_interrupted_shard_retries_and_verifies_completed_snapshot(tmp_path,monkeypatch,capsys):
    shard=tmp_path/'model-00001-of-00001.safetensors'
    shard.write_bytes(b'weights')
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps(
        dict(weight_map={'model.weight':shard.name})))
    attempts=[]
    def fetch(size,download=False):
        attempts.append((size,download))
        if len(attempts)==1:
            raise ChunkedEncodingError('interrupted')
        return tmp_path
    monkeypatch.setattr(download,'snapshot_path',fetch)
    monkeypatch.setattr(download.time,'sleep',lambda _:None)
    monkeypatch.setattr(sys,'argv',['download','--sizes','7b','--retries','2'])
    download.main()
    assert attempts==[('7b',True),('7b',True)]
    assert 'COMPLETE 7b' in capsys.readouterr().out
