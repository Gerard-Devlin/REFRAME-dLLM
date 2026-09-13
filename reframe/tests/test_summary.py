import json

from summarize import main


def test_audit_summary_exposes_errors_and_excludes_speed_claims(tmp_path, monkeypatch, capsys):
    path = tmp_path / "audit.jsonl"
    row = dict(method="pair", tiny=False, useful_tokens=4,
               stats=dict(diagnostic_run=True, elapsed_seconds=1.0, nfe=4,
                          full_forwards=2, fallback_refreshes=1,
                          fallback_reasons=["unsafe_transform_layer_2_side_1"],
                          audits=[dict(logits_error=0.1, top1_agreement=0.5,
                                       commit_set_agreement=False, committed_token_agreement=True),
                                  dict(logits_error=0.3, top1_agreement=1.0,
                                       commit_set_agreement=True, committed_token_agreement=False)]))
    path.write_text(json.dumps(row), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["summarize.py", str(path)])
    main()
    output = capsys.readouterr().out
    assert "NOT real-model acceleration evidence" in output
    assert " NA" in output
    assert "audited_decisions=2" in output and "mean_logits_error=0.200000" in output
    assert "max_logits_error=0.300000" in output
    assert "mean_top1_agreement_per_audit=0.750000" in output
    assert "commit_set_disagreements=1/2" in output
    assert "committed_token_disagreements=1/2" in output
    assert "'unsafe_transform': 1" in output
