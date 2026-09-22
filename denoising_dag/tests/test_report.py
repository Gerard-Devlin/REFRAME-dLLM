from denoising_dag.benchmark import summarize


def test_speed_is_latency_ratio_and_invalid_decisions_remove_validation():
    record=dict(depth=3,semantic_check_pass=True,logical_rows=157,unique_rows=42,
                median_wall_seconds=dict(tree=4.,reuse=2.))
    summary=summarize([record],'probe')['3']
    assert summary['validated_speedup']==2.
    assert summary['duplicate_fraction']==1-42/157
    record['semantic_check_pass']=False
    assert summarize([record],'probe')['3']['validated_speedup'] is None
