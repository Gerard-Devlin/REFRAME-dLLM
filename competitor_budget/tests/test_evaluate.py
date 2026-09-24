from competitor_budget.evaluate import method_names, method_order, summarize
from competitor_budget.report import render


def test_sweep_orders_rotate_on_each_gpu_even_when_world_matches_methods():
    names = method_names("sweep")
    world = len(names)
    for rank in range(world):
        orders = [method_order(names, rank + k * world, world) for k in range(len(names))]
        assert all(set(order) == set(names) for order in orders)
        assert {order[0] for order in orders} == set(names)


def test_sweep_reports_paired_gains_and_losses_not_just_aggregate_accuracy():
    records = []
    for index in range(4):
        record = {}
        for name in method_names("sweep"):
            record[name] = dict(correct=index < 2 if name == "official" else index > 0,
                                calls=10 if name == "official" else 5,
                                seconds=2 if name == "official" else 1,
                                length_capped=False, tokens=100)
        records.append(record)
    result = summarize(records, "sweep")
    assert result["official"]["accuracy"] == 0.5
    assert result["stable"]["accuracy"] == 0.75
    assert result["paired"]["stable"]["correct_to_wrong"] == 1
    assert result["paired"]["stable"]["wrong_to_correct"] == 2
    assert result["paired"]["stable"]["aggregate_sample_latency_speedup"] == 2.0
    table = render(dict(mode="sweep", model="synthetic", results=result))
    assert "stable" in table and "1/2" in table and "2.00x" in table
