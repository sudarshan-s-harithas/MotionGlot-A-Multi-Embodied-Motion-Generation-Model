"""Export the released checkpoint for serverless browser inference.

Run with Python 3.11, torch 2.4.1, transformers 4.44.2, numpy 1.26.4,
onnx 1.17.0 and onnxruntime 1.20.1. Models are uploaded separately from Git.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import onnx
from onnx import numpy_helper, helper
import onnxruntime as ort
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from motionglot.vq import load_vq, load_normalization
from motionglot.tasks import TASKS


class CachedGPT(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids, *past):
        cache = tuple((past[i], past[i + 1]) for i in range(0, len(past), 2))
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         position_ids=position_ids, past_key_values=cache,
                         use_cache=True, return_dict=True)
        return (out.logits[:, -1, :],) + tuple(t for pair in out.past_key_values for t in pair)


class VQPart(torch.nn.Module):
    def __init__(self, model, mode):
        super().__init__()
        self.model, self.mode = model, mode

    def forward(self, values):
        return self.model.encode(values) if self.mode == 'encoder' else self.model.forward_decoder(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint_dir / 'gpt',
                                                attn_implementation='eager').eval()
    wrapper = CachedGPT(model).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir / 'gpt')
    names = [f'past_{i}_{kind}' for i in range(12) for kind in ('key', 'value')]
    outputs = ['logits'] + [n.replace('past', 'present') for n in names]
    dummy = (torch.tensor([[1, 2, 3]]), torch.ones(1, 4, dtype=torch.long),
             torch.tensor([[1, 2, 3]])) + tuple(torch.zeros(1, 12, 1, 64) for _ in names)
    axes = {'input_ids': {1: 'sequence'}, 'attention_mask': {1: 'total_sequence'},
            'position_ids': {1: 'sequence'}}
    axes.update({n: {2: 'past_sequence'} for n in names})
    axes.update({n: {2: 'total_sequence'} for n in outputs[1:]})
    fp32 = args.output / 'gpt-fp32.onnx'
    if not (args.output / 'gpt.onnx').exists():
        print('Exporting GPT with key/value cache', flush=True)
        with torch.no_grad():
            torch.onnx.export(wrapper, dummy, fp32, input_names=['input_ids', 'attention_mask', 'position_ids'] + names,
                              output_names=outputs, dynamic_axes=axes, opset_version=17, do_constant_folding=False)
        print('Storing weights as float16, retaining float32 computation', flush=True)
        graph = onnx.load(str(fp32))
        casts = []
        for init in graph.graph.initializer:
            if init.data_type == onnx.TensorProto.FLOAT:
                original = init.name
                packed = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), original + '_packed')
                init.CopyFrom(packed)
                casts.append(helper.make_node('Cast', [packed.name], [original], to=onnx.TensorProto.FLOAT))
        nodes = list(graph.graph.node)
        del graph.graph.node[:]
        graph.graph.node.extend(casts + nodes)
        onnx.checker.check_model(graph)
        onnx.save(graph, str(args.output / 'gpt.onnx'))
        fp32.unlink()
        del graph
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(args.output / 'gpt.onnx'), options, providers=['CPUExecutionProvider'])
    prompt = 'give human motion: a person walks forward<eop><toh> \n '
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')['input_ids']
    cache = tuple(torch.zeros(1, 12, 0, 64) for _ in names)
    test_inputs = (ids, torch.ones_like(ids), torch.arange(ids.shape[1])[None]) + cache
    with torch.no_grad():
        expected = wrapper(*test_inputs)[0].numpy()
    actual = session.run(None, dict(zip(['input_ids', 'attention_mask', 'position_ids'] + names,
                                      [t.numpy() for t in test_inputs])))
    validation = {'gpt_max_logit_error': float(np.abs(actual[0] - expected).max()),
                  'gpt_mean_logit_error': float(np.abs(actual[0] - expected).mean()),
                  'first_prompt': prompt, 'first_prompt_ids': ids[0].tolist()}
    assert validation['gpt_max_logit_error'] < 0.1, validation
    # Exercise a zero-length cache and an incremental step, as the browser does.
    step = (torch.tensor([[50257]]), torch.ones(1, ids.shape[1] + 1, dtype=torch.long),
            torch.tensor([[ids.shape[1]]])) + tuple(torch.from_numpy(x) for x in actual[1:])
    session.run(None, dict(zip(['input_ids', 'attention_mask', 'position_ids'] + names, [t.numpy() for t in step])))
    del session, model, wrapper
    for embodiment, dim in [('human', 263), ('robot', 3)]:
        vq, config = load_vq(args.checkpoint_dir / f'{embodiment}_vq.pth', embodiment)
        for mode in ['encoder', 'decoder']:
            part = VQPart(vq, mode).eval()
            values = torch.randn(1, 48, dim) if mode == 'encoder' else torch.arange(12)[None]
            path = args.output / f'{embodiment}-{mode}.onnx'
            print(f'Exporting {path.name}', flush=True)
            with torch.no_grad():
                torch.onnx.export(part, (values,), path, input_names=['values'], output_names=['output'],
                                  dynamic_axes={'values': {1: 'input_frames'}, 'output': {1: 'output_frames'}},
                                  opset_version=17)
                expected = part(values).numpy()
            session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
            actual = session.run(None, {'values': values.numpy()})[0]
            error = float(np.abs(actual - expected).max())
            validation[f'{embodiment}_{mode}_max_error'] = error
            if mode == 'encoder':
                assert np.array_equal(actual, expected), (embodiment, 'encoder codes differ')
            else:
                assert error < 1e-3, (embodiment, error)
            del session
    mean, std = load_normalization(args.checkpoint_dir / 'normalization.npz')
    config = {'mean': mean.tolist(), 'std': std.tolist(), 'tasks': {k: vars(v) for k, v in TASKS.items()},
              'tokens': tokenizer.get_vocab(), 'layers': 12, 'heads': 12, 'headDim': 64, 'context': 1024}
    (args.output / 'browser-config.json').write_text(json.dumps(config))
    for name in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']:
        shutil.copyfile(args.checkpoint_dir / 'gpt' / name, args.output / name)
    (args.output / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    manifest = {'source': 'sudarshanharithas/MotionGlot@v0.1.0', 'format': 'ONNX opset 17',
                'gpt_weights': 'float16 storage; float32 computation and cache',
                'vq_weights': 'float32', 'files': {}}
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name != 'manifest.json':
            manifest['files'][path.name] = {'bytes': path.stat().st_size,
                                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == '__main__':
    main()
