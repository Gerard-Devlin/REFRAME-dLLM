"""Keep every logical history; share only deterministic predictor results.

This is the supplied proposal's bounded search, not a LoPA/MCTS reproduction.
The predictor returns sufficient statistics for this greedy, entropy-scored
policy. It cannot be substituted for a stochastic full-vocabulary sampler.
"""
from dataclasses import dataclass, field
import hashlib
import math
import time
from typing import Callable


@dataclass(frozen=True)
class Context:
    # A new immutable snapshot ID is minted after each prompt/cache change.
    snapshot_id: str
    model_revision: str
    positions: tuple[int, ...]
    attention: str
    noise_time: str = 'none'
    mode: str = 'eval-bf16-sdpa'


@dataclass(frozen=True)
class Prediction:
    token: tuple[int, ...]
    logp: tuple[float, ...]
    entropy: tuple[float, ...]


class Executor:
    def __init__(self, context: Context, predictor: Callable, reuse: bool, max_entries=4096):
        self.context, self.predictor, self.reuse = context, predictor, reuse
        self.max_entries = max_entries
        self.table = {}
        self.logical_rows = self.physical_rows = self.hits = 0
        self.key_seconds = self.dispatch_seconds = 0.
        self.layers = []

    def evaluate(self, states):
        self.logical_rows += len(states)
        if not self.reuse:
            result = self.predictor(states)
            if len(result) != len(states):
                raise ValueError('Predictor row count differs')
            self.physical_rows += len(states)
            self.layers.append(dict(logical=len(states),physical=len(states),hits=0))
            return result
        start = time.perf_counter()
        # Python dictionaries compare complete keys after a hash match.
        # Token VALUES are part of the tuple, not merely mask positions.
        keys = [(self.context, tuple(x)) for x in states]
        missing = dict.fromkeys(k for k in keys if k not in self.table)
        self.key_seconds += time.perf_counter()-start
        fresh = self.predictor([k[1] for k in missing]) if missing else []
        if len(fresh) != len(missing):
            raise ValueError('Predictor row count differs')
        self.physical_rows += len(missing)
        self.hits += len(states)-len(missing)
        start = time.perf_counter()
        local = dict(zip(missing,fresh))
        result = [self.table[k] if k in self.table else local[k] for k in keys]
        for k,v in local.items():
            if len(self.table) < self.max_entries:
                self.table[k] = v
        self.dispatch_seconds += time.perf_counter()-start
        self.layers.append(dict(logical=len(states),physical=len(missing),hits=len(states)-len(missing)))
        return result


@dataclass
class Node:
    state: tuple[int, ...]
    path: tuple[tuple[int, int], ...] = ()
    edge_logp: tuple[float, ...] = ()


@dataclass
class SearchResult:
    state: tuple[int, ...]
    path: tuple[tuple[int, int], ...]
    score: float
    decisions: str
    visits: int
    trace: list = field(default_factory=list)


def search(root, executor, mask_id, depth=3, width=4, max_nodes=4096, trace=False):
    if depth<1 or width<1 or max_nodes<1:
        raise ValueError('Positive depth, width and node budget required')
    theoretical = sum(width**i for i in range(depth+1))
    if theoretical>max_nodes:
        raise ValueError(f'Search would allow {theoretical} nodes, exceeds {max_nodes}')
    frontier = [Node(tuple(root))]
    terminals, records = [], []
    digest = hashlib.sha256()
    visits = 0
    for level in range(depth+1):
        active=[node.state for node in frontier if mask_id in node.state]
        predictions=iter(executor.evaluate(active) if active else [])
        children = []
        for node in frontier:
            unknown = [i for i,t in enumerate(node.state) if t==mask_id]
            pred=next(predictions) if unknown else None
            if pred is not None and (len(pred.token)!=len(node.state) or len(pred.logp)!=len(node.state) or len(pred.entropy)!=len(node.state)):
                raise ValueError('Prediction shape differs')
            visits += 1
            if trace and pred is not None:
                records.append((level,node.state,pred))
            if level==depth or not unknown:
                terminal = -math.fsum(pred.entropy[i] for i in unknown)/len(unknown) if unknown else 0.
                score = math.fsum((*node.edge_logp,terminal))
                terminals.append((score,node))
                digest.update(repr(('leaf',node.path,node.state)).encode())
                continue
            order = sorted(unknown,key=lambda i:(-pred.logp[i],i))[:width]
            actions = tuple((i,pred.token[i]) for i in order)
            digest.update(repr((node.path,node.state,actions)).encode())
            for i,token in actions:
                if token==mask_id or not math.isfinite(pred.logp[i]):
                    raise ValueError('Invalid greedy candidate')
                state=list(node.state)
                state[i]=token
                children.append(Node(tuple(state),node.path+((i,token),),node.edge_logp+(pred.logp[i],)))
        frontier = children
        if not frontier:
            break
    score,node = min(terminals,key=lambda pair:(-pair[0],pair[1].path))
    digest.update(repr(('selected',node.path,node.state)).encode())
    return SearchResult(node.state,node.path,score,digest.hexdigest(),visits,records)


def compare(reference, candidate, mask_id=-1):
    same_graph = [(a,x) for a,x,_ in reference.trace]==[(a,x) for a,x,_ in candidate.trace]
    token_flips, logp_error, entropy_error = 0,0.,0.
    if same_graph:
        for (_,state,a),(_,_,b) in zip(reference.trace,candidate.trace):
            used=[i for i,t in enumerate(state) if t==mask_id]
            token_flips += sum(a.token[i]!=b.token[i] for i in used)
            logp_error=max(logp_error,max(abs(a.logp[i]-b.logp[i]) for i in used))
            entropy_error=max(entropy_error,max(abs(a.entropy[i]-b.entropy[i]) for i in used))
    else:
        token_flips=logp_error=entropy_error=None
    same_decisions = reference.decisions==candidate.decisions
    return dict(same_graph=same_graph,same_decisions=same_decisions,
                same_selected_path=reference.path==candidate.path,
                same_endpoint=reference.state==candidate.state,
                selected_score_abs_error=abs(reference.score-candidate.score),
                policy_top1_flips=token_flips,policy_logp_max_abs=logp_error,
                policy_entropy_max_abs=entropy_error,
                pass_=same_graph and same_decisions and token_flips==0)
