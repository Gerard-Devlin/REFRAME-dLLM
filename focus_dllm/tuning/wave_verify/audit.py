"""CPU/source audit of the user's exact-graph lazy-verification proposal."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from .graph import analyze


def pinned_topology(source):
    tree=ast.parse(source)
    expected=(
        'm11 = indices.view(-1, 1) >= indices.view(1, -1)',
        'm12 = indices.view(-1, 1) < indices.view(1, -1)',
        'm21 = indices.view(-1, 1) > indices.view(1, -1)',
        'm22 = indices.view(-1, 1) <= indices.view(1, -1)',
        'causal_mask[:T, T:T+S] = False',
        'causal_mask[T:T+S+S, T:T+S+S] = torch.cat([torch.cat([m11, m12], dim=1), torch.cat([m21, m22], dim=1)], dim=0)',
    )
    actual=[ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.Assign)]
    for assignment in expected:
        assert actual.count(ast.dump(ast.parse(assignment).body[0],include_attributes=False))==1,assignment
    return {'exact_source_expressions_checked':len(expected),'residual_edges_included':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--diagnostic',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=dict(source=str(args.source),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        topology=pinned_topology(args.source.read_text(encoding='utf-8')),
        geometry=[analyze(search=s,prefix=p) for s in (4,8,16) for p in (1,s)],
        outcome='No deep prefix-only pruning for the unchanged pinned graph; no GPU scheduler deployed',
        scope='CPU mathematical/source audit, not measured latency, quality or a universal impossibility result')
    if args.diagnostic:
        raw=json.loads(args.diagnostic.read_text(encoding='utf-8-sig'))
        report['existing_diagnostic_sha256']=hashlib.sha256(args.diagnostic.read_bytes()).hexdigest()
        rows=raw.get('diagnostic',raw)['controls']
        report['existing_windows']=dict(count=len(rows),
            with_rejected_tail=sum(c['official_cumulative_accepted']<c['search'] for c in rows),
            conditional_high_tail=sum(c['official_high_tail_after_first_failure'] for c in rows),
            scope='Selected development windows; tail predictions depend on rejected drafts and are not free savings')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'outcome':report['outcome'],'rows':report['geometry'][-2]['required_input_rows']}))


if __name__=='__main__':main()
