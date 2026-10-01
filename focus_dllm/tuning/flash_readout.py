"""Consumed-row LM head adapter; original Flash decoding/cache remain intact.

The official generator is cloned locally to pass its ALREADY KNOWN Python row
offsets. No new GPU synchronization, source edit, or token-selection rule.
BF16 head-shape effects still require measured action regression.
"""
import ast
from contextlib import contextmanager
import functools
import inspect
import io
import textwrap

import torch

from .padded_head import pad_rows


class Readout:
    """Virtual full-row indexing backed ONLY by the consumed compact rows."""
    def __init__(self, logits, start, count, full_rows):
        if logits.ndim!=3 or logits.shape[0]!=1 or logits.shape[1]<count:
            raise ValueError('Batch-one compact head required')
        if start<0 or count<0 or start+count>full_rows:raise ValueError('Invalid head row range')
        self.logits=logits[0,:count];self.start=start;self.count=count;self.full_rows=full_rows

    def squeeze(self, dim):
        if dim!=0:raise ValueError('Only the pinned generator squeeze(0) is supported')
        return self

    def __getitem__(self, key):
        if not isinstance(key,slice) or key.step not in (None,1):
            raise ValueError('Only consumed contiguous slices are supported')
        start=0 if key.start is None else key.start
        stop=self.full_rows if key.stop is None else key.stop
        if not self.start<=start<=stop<=self.start+self.count:
            raise ValueError('Generator attempted to read an unprojected row')
        return self.logits[start-self.start:stop-self.start]


class ModelReadout:
    def __init__(self, model, *, compact=False, minimum=32, observer=None):
        self.model_ref=model;self.compact=compact;self.minimum=minimum;self.observer=observer
        self.current=None

    def __getattr__(self,name):return getattr(self.model_ref,name)

    def __call__(self,*args,focus_head_rows=None,**kwargs):
        if focus_head_rows is None:raise ValueError('Missing pinned consumer row metadata')
        start,count=map(int,focus_head_rows)
        self.current=dict(start=start,count=count,verify=bool(kwargs['lengths'][-1]))
        if not self.compact:
            value=self.model_ref(*args,**kwargs)
            if self.observer:self.observer(self.current,args,kwargs,value)
            return value
        full=[None]
        def select(_module,_inputs,value):
            full[0]=value.shape[1]
            if not 0<=start<=start+count<=full[0]:raise ValueError('Head consumer outside hidden canvas')
            # Preserve FULL final normalization, then reduce only projection rows.
            # Empty verification has no consumer; Transformer/cache still execute.
            return pad_rows(value[:,start:start+count],self.minimum if count else 0)
        norm=self.model_ref.model.transformer.ln_f
        handle=norm.register_forward_hook(select)
        try:value=self.model_ref(*args,**kwargs)
        finally:handle.remove()
        from types import SimpleNamespace
        return SimpleNamespace(logits=Readout(value.logits,start,count,full[0]))


def generator(function, action_observer=None, *, statistics=None):
    """Fail closed on unknown source; change only consumer metadata/telemetry."""
    original=inspect.unwrap(function)
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    if len(functions)!=1:raise ValueError('Expected one pinned generator')
    functions[0].decorator_list=[]
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='model']
    calls.sort(key=lambda n:n.lineno)
    if len(calls)!=2:raise ValueError('Expected exactly normal and verify model calls')
    expressions=('(0, block_m) if verify else (0, query_masked_pos[0].shape[0])',
                 '(seqlen_keep[0] + num_verify, num_verify)')
    for call,expression in zip(calls,expressions):
        if any(k.arg=='focus_head_rows' for k in call.keywords):raise ValueError('Metadata collision')
        call.keywords.append(ast.keyword(arg='focus_head_rows',value=ast.parse(expression,mode='eval').body))
    additions=[0]
    reductions=dict(masked=0,verify=0,probability=0)
    class Actions(ast.NodeTransformer):
        def visit_Assign(self,node):
            self.generic_visit(node)
            if statistics is not None:
                names=[n.id for n in ast.walk(ast.Tuple(elts=node.targets,ctx=ast.Load()))
                       if isinstance(n,ast.Name)]
                def expected(source):
                    return ast.dump(node,include_attributes=False)==ast.dump(ast.parse(source).body[0],include_attributes=False)
                if names in (['p_masked'],['p_verify']):
                    stem='masked' if names==['p_masked'] else 'verify'
                    if not expected(f'p_{stem} = F.softmax(logits_{stem}_j.to(torch.float64), dim=-1)'):
                        raise ValueError('Unexpected full-vocabulary probability source')
                    reductions['probability']+=1
                    return None
                if names==['x0_p_masked','x0_masked']:
                    if not expected('x0_p_masked, x0_masked = torch.max(p_masked, dim=-1)'):
                        raise ValueError('Unexpected normal confidence source')
                    reductions['masked']+=1
                    node.value=ast.parse('_focus_statistics(logits_masked_j, None)',mode='eval').body
                elif names==['x0_p_verify'] and expected('x0_p_verify = p_verify.gather(1, x_verify_j.unsqueeze(1)).view(-1)'):
                    reductions['verify']+=1
                    node.value=ast.parse('_focus_statistics(logits_verify_j, x_verify_j)[0]',mode='eval').body
            if action_observer is None:return node
            if len(node.targets)!=1:return node
            target=node.targets[0]
            if not (isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) and target.value.id=='x'
                    and isinstance(target.slice,ast.Name) and target.slice.id=='pos_decoded_new_j'):return node
            if not isinstance(node.value,ast.Name) or node.value.id!='x0_decoded_new_j':
                raise ValueError('Unexpected commit source')
            additions[0]+=1
            callback=ast.parse('_focus_action(pos_decoded_new_j, x0_decoded_new_j)').body[0]
            return [node,ast.copy_location(callback,node)]
    tree=Actions().visit(tree)
    if action_observer is not None and additions[0]!=2:raise ValueError('Expected two actual canvas commit sites')
    if statistics is not None:
        if reductions!=dict(masked=2,verify=1,probability=3):
            raise ValueError('Expected exactly three pinned probability consumers')
        if any(isinstance(n,ast.Name) and n.id in ('p_masked','p_verify') for n in ast.walk(tree)):
            raise ValueError('Additional full distribution consumers cannot be bypassed')
    ast.fix_missing_locations(tree)
    scope=dict(original.__globals__)
    if '_focus_action' in scope:raise ValueError('Observer namespace collision')
    scope['_focus_action']=action_observer
    if '_focus_statistics' in scope:raise ValueError('Statistics namespace collision')
    scope['_focus_statistics']=statistics
    exec(compile(tree,original.__code__.co_filename+':consumed_head_rows','exec'),scope)
    return functools.update_wrapper(torch.no_grad()(scope[original.__name__]),original)


@contextmanager
def suppress_official_prints():
    from contextlib import redirect_stdout
    with redirect_stdout(io.StringIO()):yield
