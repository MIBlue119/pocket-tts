"""Bounded diagnostic harness; preserve normal evaluator code and immutable outputs."""
import argparse
import functools
import json
from pathlib import Path
from training.taigi import evaluate as evaluator
from training.taigi.audio import sha256_file

p = argparse.ArgumentParser()
p.add_argument('--config', required=True, type=Path)
p.add_argument('--checkpoint', required=True, type=Path)
p.add_argument('--prompts', required=True, type=Path)
p.add_argument('--output', required=True, type=Path)
p.add_argument('--decode-steps', required=True, type=int, choices=[1, 4])
a = p.parse_args()
original_build = evaluator.build_models

def build_with_explicit_decode_steps(config):
    model, mimi, auxiliary = original_build(config)
    model.generate = functools.partial(model.generate, n_steps=a.decode_steps)
    return model, mimi, auxiliary

evaluator.build_models = build_with_explicit_decode_steps
report = evaluator.evaluate(argparse.Namespace(config=a.config, checkpoint=a.checkpoint,
    prompts=a.prompts, output=a.output, device='cuda:0', max_frames=100, seed=17,
    cfg_coef=1.0, asr=True))
x = json.loads(report.read_text())
x.update(sampler_decode_steps=a.decode_steps, diagnostic_only=True,
         diagnostic_harness_sha256=sha256_file(Path(__file__)),
         generation_override='Process-local functools.partial passes n_steps explicitly to TrainableTTS.generate')
report.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
print(report)
