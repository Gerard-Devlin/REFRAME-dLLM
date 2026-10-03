"""Fail before loading weights if a guarded research launch lost its binding."""
import os


def check_binding(required=False):
    expected=os.environ.get('FOCUS_RESEARCH_GPU_UUID')
    if expected is None:
        if required:
            raise RuntimeError('Missing explicit research GPU UUID')
        return None
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible!=expected or not expected.startswith('GPU-'):
        raise RuntimeError('Research GPU binding was lost before model loading')
    import torch
    if torch.cuda.device_count()!=1:
        raise RuntimeError('Research process must see exactly one GPU')
    props=torch.cuda.get_device_properties(0)
    actual=str(getattr(props,'uuid',expected))
    # Older PyTorch does not expose uuid; the exact CUDA_VISIBLE_DEVICES UUID
    # remains the CUDA device selection contract in that case.
    if actual.lower().removeprefix('gpu-')!=expected.lower().removeprefix('gpu-'):
        raise RuntimeError('CUDA resolved a different physical GPU')
    return dict(visible_uuid=visible,actual_uuid=actual,device_name=props.name)
