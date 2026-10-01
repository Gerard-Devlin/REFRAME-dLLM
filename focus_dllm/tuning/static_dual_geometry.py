"""Temporary native DualCache index hoisting with fixed-address position data.

Only the pinned native attention/RoPE metadata assignments are replaced. The
tensor arithmetic, cache writes, attention and projections remain native. This
is a common execution control, not an algorithmic acceleration or certificate.
"""
import ast
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import textwrap
import types

import torch


NATIVE_LF_SHA256='c60ed78a8f044c885f49c1ba0685b3cbb9c697232b4da85844c4a9cebc426459'


def compile_native(function,kind):
    source=Path(inspect.getsourcefile(function))
    if hashlib.sha256(source.read_text(encoding='utf-8').encode()).hexdigest()!=NATIVE_LF_SHA256:
        raise ValueError('Native implementation differs from the audited source')
    tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
    definition=tree.body[0]
    expected=({'batch_replace_indices':'replace_position[batch_idx].nonzero(as_tuple=True)[0]',
        'max_replace_pos':'replace_position.nonzero(as_tuple=True)[1].max() + 1 if replace_position.any() else key_len'}
        if kind=='attention' else {'idx':'torch.arange(start, end, device=q_.device, dtype=torch.long)'})
    replacements=({'batch_replace_indices':'self._focus_static_geometry.indices[batch_idx]',
        'max_replace_pos':'self._focus_static_geometry.end'} if kind=='attention' else
        {'idx':'self._focus_static_geometry.indices[0]'})
    found=[]

    class Hoist(ast.NodeTransformer):
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                name=node.targets[0].id
                if name in expected:
                    if ast.dump(node.value)!=ast.dump(ast.parse(expected[name],mode='eval').body):
                        raise ValueError('Unexpected native metadata expression')
                    node.value=ast.parse(replacements[name],mode='eval').body
                    found.append(name)
            return self.generic_visit(node)

    Hoist().visit(definition)
    if sorted(found)!=sorted(expected):
        raise ValueError('Missing/duplicated native metadata expression')
    definition.name='_focus_'+kind
    if kind=='attention':
        definition.body.insert(0,ast.parse(
            'self._focus_static_geometry.check_call(q, layer_past, replace_position)').body[0])
    namespace=function.__globals__.copy()
    exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),namespace)
    return namespace[definition.name]


class StaticDualGeometry:
    def __init__(self,mask):
        if mask.ndim!=2 or mask.dtype!=torch.bool or not mask.shape[0]:
            raise ValueError('Batched boolean replacement mask required')
        cpu=mask.detach().cpu()
        indices=cpu[0].nonzero().flatten()
        if not indices.numel() or not torch.equal(indices,torch.arange(int(indices[0]),int(indices[-1])+1)):
            raise ValueError('One nonempty contiguous native block required')
        if not all(torch.equal(row,cpu[0]) for row in cpu):
            raise ValueError('All verification rows must share one block')
        self.width,self.length=mask.shape
        self.query=indices.numel()
        # Own all writable buffers; the original teacher mask is never changed.
        self.mask=mask.detach().clone()
        shared=indices.to(mask.device)
        self.indices=[shared]*self.width
        self.end=int(indices[-1])+1
        self.saved=[]
        self.expected_versions=(self.mask._version,shared._version)
        self.addresses=(self.mask.data_ptr(),shared.data_ptr())

    def validate(self):
        if self.addresses!=(self.mask.data_ptr(),self.indices[0].data_ptr()):
            raise ValueError('Graph geometry storage changed')
        if self.expected_versions!=(self.mask._version,self.indices[0]._version):
            raise ValueError('Geometry changed outside the explicit block update')

    def set_block(self,start):
        if not isinstance(start,int) or not 0<=start<=self.length-self.query:
            raise ValueError('Replacement block is outside the canvas')
        self.validate()
        if self.mask.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Geometry preparation must be paid outside capture')
        fresh=torch.zeros_like(self.mask);fresh[:,start:start+self.query]=True
        self.mask.copy_(fresh)
        self.indices[0].copy_(torch.arange(start,start+self.query,device=self.mask.device))
        self.end=start+self.query
        self.expected_versions=(self.mask._version,self.indices[0]._version)

    def check_call(self,q,past,mask):
        self.validate()
        if mask is not self.mask or q.ndim!=3 or tuple(q.shape[:2])!=(self.width,self.query):
            raise ValueError('Forward does not match the prepared block/rows')
        if past is None or len(past)!=2:
            raise ValueError('Only ordinary native DualCache calls are supported')
        for value in past:
            if value.ndim!=4 or value.shape[0]!=self.width or value.shape[-2]!=self.length:
                raise ValueError('Cache shape differs from the prepared canvas')
            if self.width>1 and value.stride(0)<value[0].numel():
                raise ValueError('Writable cache rows overlap or alias')

    def install(self,obj,name,value):
        self.saved.append((obj,name,name in obj.__dict__,obj.__dict__.get(name)))
        setattr(obj,name,value)

    @contextmanager
    def scope(self,model):
        if self.saved:
            raise RuntimeError('Nested geometry installation is unsupported')
        compiled={}
        try:
            for block in model.model.transformer.blocks:
                for obj,name,kind in ((block,'attention','attention'),(block.rotary_emb,'forward','rotary')):
                    original=getattr(obj,name).__func__
                    if (original,kind) not in compiled:
                        compiled[original,kind]=compile_native(original,kind)
                    self.install(obj,'_focus_static_geometry',self)
                    self.install(obj,name,types.MethodType(compiled[original,kind],obj))
            yield self
        finally:
            for obj,name,existed,value in reversed(self.saved):
                if existed:setattr(obj,name,value)
                else:delattr(obj,name)
            self.saved.clear()
