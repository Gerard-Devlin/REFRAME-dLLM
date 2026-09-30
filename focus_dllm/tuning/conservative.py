"""Conservative refresh and small live-support ablations on development prompts."""
from functools import partial

from . import stream_sweep
from .static_support import StaticSupportForward
from .streaming import generate_stream

run = stream_sweep.fast_zero.optimizations.experimental.run
run.VARIANTS.update({
    'stream2_l8': dict(kind='stream',layer=8,keep=0.,threshold=.90,refresh=2),
    'stream4_l8': dict(kind='stream',layer=8,keep=0.,threshold=.90,refresh=4),
    'stream4_l4_t95': dict(kind='stream',layer=4,keep=0.,threshold=.95,refresh=4),
    'support16_r4': dict(kind='support',layer=4,keep=0.,threshold=.90,refresh=4,support=16),
    'support16_r2': dict(kind='support',layer=4,keep=0.,threshold=.90,refresh=2,support=16),
})
original = stream_sweep.generate_stream


def run_one(model, tokenizer, sample, config, gen_length):
    if config['kind'] != 'support':
        return stream_sweep.run_one(model,tokenizer,sample,config,gen_length)
    options = dict(config,kind='stream')
    def generate(*args, **kwargs):
        return generate_stream(*args,**kwargs,
            forward_type=partial(StaticSupportForward,support_count=config['support']))
    stream_sweep.generate_stream = generate
    try:
        return stream_sweep.run_one(model,tokenizer,sample,options,gen_length)
    finally:
        stream_sweep.generate_stream = original


run.run_one = run_one

if __name__ == '__main__':
    run.main()
