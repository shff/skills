# SKILL: .ckpt → CoreML for audio models

Proven path for converting PyTorch `.ckpt` audio models (MDX separation,
transformer beat trackers, RoFormer-style architectures) to a CoreML
`.mlpackage` that runs on Apple Silicon.

Two real conversions documented here:

| Model | Architecture | Result |
|---|---|---|
| `config_mdx23c_similarity.ckpt` | TFC-TDF-Net + complex STFT | 7-min WAV in 18 s (~23× RT) on ANE, null-clean |
| `final0.ckpt` (Beat This!) | RoFormer + log-mel input | 30-s chunk in 20 ms on CPU+GPU (~1450× RT) |

---

## Prerequisites

```bash
# Python 3.11 only — coremltools doesn't support 3.12+
python3.11 -m venv coreml/.venv
source coreml/.venv/bin/activate
pip install coremltools==9.0 "torch==2.6.0" "torchaudio==2.6.0" numpy
# Plus whatever the model's inference code imports (einops, rotary_embedding_torch,
# soxr, librosa, soundfile, etc.)
```

Notes:
- `coremltools==9.0` works with `torch==2.6.0`. Torch and torchaudio versions
  **must match** or torchaudio's `.so` will fail to dlopen.
- coremltools may warn about untested torch versions — usually fine, but watch
  for `aten::Int` errors (see Step 2).

---

## Step 1 — Read the Python inference code

Before writing any conversion code, identify all of these. Each one is a
landmine that breaks `torch.jit.trace` → CoreML conversion:

| Pattern in source | Why it breaks CoreML | Fix |
|---|---|---|
| `torch.stft` / `torch.istft` | MIL IR has no complex ops | Replace with `F.conv1d` using sin/cos basis |
| `b, c, f, t = x.shape` then reshape | Generates `aten::Int` | Hardcode dims (input shape is fixed anyway) |
| `b = len(x)` / `b = x.shape[0]` used in `rearrange(..., b=b)` | Generates `aten::Int` via einops | Use static Python ints from weight shapes |
| `einops.layers.torch.Rearrange` in `nn.Sequential` | Internal `aten::Int` ops | Replace with `nn.Module` using `permute`+`reshape` |
| `einops.rearrange(...)` in `forward()` | Same | Same — use `permute`+`reshape`/`view` |
| `rotary_embedding_torch.rotate_queries_or_keys` | Dynamic `rot_dim`, dynamic einops | Precompute cos/sin buffers, write `_static_rotate` |
| Dict output (`return {"a": x, "b": y}`) | `torch.jit.trace` rejects | Return a `tuple`; CoreML flattens it |
| `torch.autocast(...)` w/ device-type check | Tracer can't follow Python conditions | Plain `.float()` cast |
| `q, k, v = rearrange(...)` (tensor iteration) | TracerWarning + `aten::Int` | Use `tensor.split(dim_inner, dim=-1)` |
| `rotary_embedding_torch` cache logic | Bool-from-tensor conditions | Bypass entirely with precomputed buffers |

For each model, find the fixed input shape too — every conversion below
assumes it.

| Model | Input | Output |
|---|---|---|
| MDX TFC-TDF-Net | audio chunk `(1, 2, 130560)` → spec `(1, 4, 1024, 256)` | same shape |
| Beat This! BeatThis | log-mel `(1, 1500, 128)` | two `(1, 1500)` tensors |

---

## Step 2 — Understand the root cause of `aten::Int`

Almost every conversion failure for this class of model comes down to one bug
in coremltools 9.0: when a Python int is extracted from a tensor shape and
then passed to `reshape`/`view`, the trace emits `prim::NumToTensor` →
`aten::Int`, and coremltools tries to constant-fold it with
`int(numpy_0d_array)` — which raises:

```
TypeError: only 0-dimensional arrays can be converted to Python scalars
```

The fix is always the same: **never let a shape value flow through Python.**
Use one of these instead:

1. **Closure-captured Python ints.** Compute `heads`, `dim_head`, `n`, etc.
   from weight shapes at conversion time (before `trace`). Capture them in a
   closure when patching `forward`. The traced graph sees them as constants.
2. **`-1` as the single unknown dim.** `reshape(-1, n, h, d)` is fine — only
   one `-1` per reshape, and the others must be Python int constants.
3. **Precomputed buffers.** For RoPE: precompute cos/sin tables on the
   *unpatched* model, register them as buffers, then use them in a static
   `_rotate` function that takes only Python int constants for shape.

Diagnose which op is to blame by tracing a single submodule and grepping the
graph (see `coreml/debug_attn.py` from the Beat This! conversion):

```python
traced = torch.jit.trace(submodule, dummy, check_trace=False)
for node in traced.graph.nodes():
    if 'Int' in node.kind():
        print(node)
        for inp in node.inputs():
            print('  ←', inp.node())
```

The `prim::NumToTensor` parents tell you which op generates the dynamic int.
Common culprits: `aten::floor_divide` (from `dim_inner // h`), `aten::mul`
(from einops shape arithmetic), `aten::add` (from `start_index + rot_dim` in
RoPE).

---

## Step 3 — Write a traceable wrapper

The strategy depends on whether the model has STFT or not.

### Path A — Model with STFT/iSTFT (MDX, demucs-family)

Two mandatory fixes:

**Fix A1 — Replace STFT/iSTFT with conv1d**

`torch.stft` produces complex tensors. CoreML's MIL IR has no complex ops.
Replace with `F.conv1d` using sin/cos basis weights.

```python
# ConvSTFT: basis shape (2*n_bins, 1, n_fft)
# rows 0..n_bins-1        = cos(angle) * window
# rows n_bins..2*n_bins-1 = -sin(angle) * window     ← minus matters (e^{-i2πkn/N})
F.conv1d(reflect_pad(x), basis, stride=hop_length)

# ConvISTFT: same basis, scaled — 2/N for bins [1..N/2-1], 1/N for DC and Nyquist
# Then divide by window-squared OLA envelope
F.conv_transpose1d(x, basis_scaled, stride=hop_length)
```

Numerical check: max relative error vs `torch.stft` ~1.85e-4 (FP32 round-trip
noise).

**Fix A2 — Hardcode dynamic shapes**

```python
# bad — emits aten::Int
b, c, f, t = x.shape
x = x.reshape(b, c * k, f // k, t)

# good — static reshape, input is always (1, 4, 1024, 256) anyway
x = x.reshape(1, 16, 256, 256)
```

**Spec-only vs full-graph variants**

| File | Wraps | Use case |
|---|---|---|
| `model_traceable.py` | audio in → audio out (conv STFT baked in) | Simple Swift host, no vDSP |
| `model_spec_traceable.py` | spec in → spec out | **Fast path** — STFT done host-side in vDSP |

Always prefer the spec-only variant. The conv STFT layer has a 2049-tap
kernel that forces ANE→CPU transfers (see Step 5).

### Path B — Model with no STFT but transformer/RoFormer internals (Beat This!)

No STFT issues, but the einops + RoPE + dynamic-batch-from-shape patterns are
just as fatal. The Beat This! conversion needed five separate patches:

1. **`PartialRoformer.forward` and `PartialFTTransformer.forward`**:
   replace einops `rearrange(..., b=b)` with plain `permute`+`reshape` and
   hardcode `batch=1`. Inside these, the "effective batch" passed to attention
   is actually `t` or `f` (1500 or 32), not 1 — record what it is per
   module.

2. **`roformer.Attention.forward`**: replace the
   `rearrange("b n (qkv h d) -> qkv b h n d")` with `qkv.split(dim_inner, 2)`
   and explicit reshapes using **closure-captured Python ints**:

   ```python
   def _make_attention_forward(heads, dim_head, n_static, b_static, ...):
       dim_inner = heads * dim_head
       def forward(self, x):
           x = self.norm(x)
           q, k, v = self.to_qkv(x).split(dim_inner, dim=2)
           q = q.reshape(b_static, n_static, heads, dim_head).permute(0, 2, 1, 3)
           # ...
           out = out.permute(0, 2, 1, 3).reshape(b_static, n_static, dim_inner)
           return self.to_out(out)
       return forward
   ```

   The static sizes are recorded via a forward-hook probe pass before
   patching (see Step 3.5 below).

3. **`rotary_embedding_torch.rotate_queries_or_keys`**: don't try to fix it —
   bypass it. Precompute cos/sin tables matching its interleaved convention,
   attach them as buffers, write a static replacement:

   ```python
   def _static_rotate(t, cos, sin, b, heads, n, dim_head):
       half = dim_head // 2
       t_pairs = t.reshape(b, heads, n, half, 2)
       x1 = t_pairs[..., 0]
       x2 = t_pairs[..., 1]
       rotated = torch.stack((-x2, x1), dim=-1).reshape(b, heads, n, dim_head)
       return t * cos + rotated * sin
   ```

   Build cos/sin by calling the *original* `rotary_embed.forward(seq, seq_len)`
   on a static seq, then `.cos()` / `.sin()`. Match the convention exactly:
   `rotary_embedding_torch.rotate_half` uses **interleaved pairs**
   (`reshape(..., D/2, 2)`), not split-in-half.

4. **`SumHead.forward`**: returns a dict → tracer rejects. Replace with a
   tuple, and replace `rearrange("b t c -> c b t", c=2)` with `permute`
   + indexing (the rearrange-then-unbind iterates a tensor, which warns and
   often misfires):

   ```python
   def _sum_head_forward(self, x):
       beat_downbeat = self.beat_downbeat_lin(x).permute(2, 0, 1)  # (2, b, t)
       beat = beat_downbeat[0]
       downbeat = beat_downbeat[1]
       beat = beat.float() + downbeat.float()
       return beat, downbeat
   ```

5. **`einops.layers.torch.Rearrange` in `nn.Sequential`**: walk the model
   and replace each with a hand-written `nn.Module` whose `forward` uses only
   `permute`/`reshape`/`unsqueeze` with static dims. Patterns seen in Beat
   This!:

   ```
   "b t f -> b f t"        →  x.permute(0, 2, 1)
   "b f t -> b 1 f t"      →  x.unsqueeze(1)
   "b c f t -> b t (c f)"  →  x.permute(0, 3, 1, 2).reshape(1, t, c*f)   # c*f static
   ```

### Step 3.5 — Probe pass for static shapes (Path B only)

Per-module attention sees different `(b, n)` depending on where it is in the
network. Don't guess — record them with a forward hook:

```python
shape_map = {}
def make_hook(name):
    def hook(module, inputs, output):
        shape_map[name] = tuple(inputs[0].shape)
    return hook
handles = [m.register_forward_hook(make_hook(name))
           for name, m in model.named_modules() if isinstance(m, Attention)]
with torch.no_grad():
    model(dummy_input)        # run with already-patched non-attention forwards
for h in handles: h.remove()
```

Then close `_make_attention_forward` over `shape_map[name]` per module.

Eager probing works with dynamic ops; you only need static shapes at
**trace** time. So order of operations is: patch non-attention forwards →
probe to record shapes → patch attention with static sizes → trace.

---

## Step 4 — Trace and convert

```python
import coremltools as ct
import numpy as np
import torch

wrapper = TraceableWrapper("path/to/model.ckpt", dummy).eval()
dummy = torch.randn(*input_shape)

with torch.no_grad():
    traced = torch.jit.trace(wrapper, dummy, check_trace=False)

# Sanity: traced must match eager bitwise (no shape drift in patches)
with torch.no_grad():
    assert (traced(dummy)[0] - wrapper(dummy)[0]).abs().max() < 1e-4

mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="input", shape=dummy.shape, dtype=np.float32)],
    outputs=[
        ct.TensorType(name="output_a", dtype=np.float32),
        ct.TensorType(name="output_b", dtype=np.float32),   # for multi-output
    ],
    compute_units=ct.ComputeUnit.ALL,
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.macOS14,
    convert_to="mlprogram",
)
mlmodel.save("model.mlpackage")
```

- Conversion takes ~5 s once the model is traceable.
- FP16 shrinks: 437 MB → 225 MB (MDX); 81 MB → ~40 MB (Beat This!).
- **Do not use the ONNX intermediate path** — hangs/fails for STFT models,
  and loses no information for non-STFT models.
- "Tuple detected at graph output. This will be flattened in the converted
  model." is informational — coremltools maps the tuple to your named
  outputs in order.

---

## Step 5 — Profile all compute units (do not skip)

ANE is **not** automatically the fastest. The right unit depends on what
ops are in the graph. Profile all three.

```python
import coremltools as ct, time, numpy as np

for unit in [ct.ComputeUnit.ALL, ct.ComputeUnit.CPU_AND_GPU, ct.ComputeUnit.CPU_ONLY]:
    m = ct.models.MLModel("model.mlpackage", compute_units=unit)
    x = {"input": np.random.randn(*input_shape).astype(np.float32)}
    for _ in range(3): m.predict(x)   # warm up
    t0 = time.time()
    for _ in range(20): m.predict(x)
    print(unit, (time.time() - t0) / 20 * 1000, "ms/chunk")
```

**Measured results:**

| Model | ALL | CPU+GPU | CPU only |
|---|---|---|---|
| MDX full-graph (conv STFT baked in) | 3256 ms | 3368 ms | **228 ms** |
| MDX spec-only | **59 ms** | 74 ms | 228 ms |
| Beat This! (RoFormer) | 213 ms | **21 ms** | 92 ms |

**Three different winners.** Patterns:

- **MDX full-graph**: the 2049-tap conv1d STFT can't run on ANE, every
  chunk pays an ANE↔CPU round-trip. CPU-only wins by 14×. Fix: externalize
  STFT (spec-only variant).
- **MDX spec-only**: ANE handles the whole conv graph cleanly → ANE wins.
- **Beat This! RoFormer**: attention + RMSNorm + gating don't ANE-accelerate
  well; the attempt to use ANE adds overhead. CPU+GPU is 10× faster than
  ANE-enabled ALL. ANE works best for ConvNets and channel-heavy workloads,
  not for transformer attention at typical sequence lengths.

**Rule**: if ANY op in the graph can't run on ANE, every chunk pays a
round-trip penalty. Externalize ANE-hostile ops from the model entirely, or
disable ANE for that model.

In Swift:

```swift
let config = MLModelConfiguration()
config.computeUnits = .cpuAndGPU   // pick whatever profiled fastest
let model = try MyModel(configuration: config)
```

---

## Step 6 — Validate numerics

Compare CoreML output to PyTorch eager output on a realistic input. Random
`N(0, 1)` input is fine for separators but misleads for log-mel models —
real log-mel features sit around mean ~7, so use that magnitude:

```python
test_in = (np.random.randn(*input_shape).astype(np.float32) * std + mean)
with torch.no_grad():
    ref = wrapper(torch.from_numpy(test_in))
cml = mlmodel.predict({"input": test_in})

diff = np.abs(cml["output"] - ref.numpy())
print(f"max: {diff.max():.3e}, mean: {diff.mean():.3e}")
```

**Expected at FP16:**

| Model | Output range | Max abs diff | Mean abs diff |
|---|---|---|---|
| MDX spec-only | ~unit | ~1e-3 | ~1e-4 |
| Beat This! beat logits | [-1.3, 0.3] | ~1e-2 | ~2e-3 |
| Beat This! downbeat logits | [-1.7, 0.4] | ~2e-2 | ~5e-3 |

Wider tolerances are normal when output magnitudes are larger. If max diff
exceeds ~3% of the output range, suspect a missed conversion bug, not FP16
quantization. Re-run with `compute_precision=ct.precision.FLOAT32` to isolate.

---

## Step 7 — Swift host (STFT path only)

If externalizing STFT to vDSP host-side:

Compile with `-O` (mandatory, ~4× faster than `-Onone`):

```bash
swiftc -O spliff2.swift -o spliff2
```

### vDSP STFT gotchas

- **Bin doubling**: `vDSP_fft_zrip` doubles all bins. Multiply forward output by 0.5.
- **Inverse scaling**: round-trip = `2*N*x[n]`. Use forward ×0.5 + inverse ×`1/(2N)`.
  Using `1/N` makes output 6 dB too quiet.
- **Nyquist packing**: vDSP packs Nyquist into `fftImag[0]`. Unpack on forward,
  repack on inverse.
- **Sign convention**: vDSP and `torch.stft` both use `e^{-i2πkn/N}` — no extra flips.
- **Window**: Hann periodic (`torch.hann_window(periodic=True)`). Apply on synthesis.
- **Center padding**: reflect-pad by N/2 each side before framing; strip N/2 after iSTFT.

### Overlap-add chunking (separation models)

- `chunk_size = 130560` (~2.96 s at 44.1 kHz)
- `num_overlap = 2` → 50% overlap, 4 STFT frames overlap per output sample
- Fade window: linear fade-in/out over `chunk_size / 10` samples at chunk edges
  (no fade-in on first chunk, no fade-out on last)
- Reflect-pad whole signal by `border = chunk_size - step` samples on each side;
  strip after reconstruction
- `num_overlap = 1` is ~2× faster but leaves boundary artifacts every 2.96 s

### Audio I/O

Use `AVAudioFile` + `AVAudioConverter` for read + resample. Write Float32 WAV.
This matches librosa's behavior closely enough for null tests to pass.

---

## Step 8 — Null-test the Swift CLI (STFT path only)

```bash
python3 inference path/to/audio.wav ref_out.wav
./spliff2 path/to/audio.wav swift_out.wav

python3 -c "
import soundfile as sf, numpy as np
a, _ = sf.read('ref_out.wav')
b, _ = sf.read('swift_out.wav')
d = np.abs(a - b)
print(f'max diff: {d.max():.3e}, mean: {d.mean():.3e}')
"
```

Passing: max diff < 1e-3. If it fails, check: scale factors, OLA window,
padding, Nyquist packing.

---

## Performance knobs (ranked by ROI)

1. **Compute unit selection** — profile everything, don't trust ALL (10×+)
2. **Externalize ANE-hostile ops** — spec-only model + vDSP for separators (4×)
3. **`-O` swiftc flag** — always (4×)
4. **`num_overlap`** — 2→1 doubles speed, adds boundary artifacts (2×)
5. **Concurrent chunks** — pipeline vDSP + ANE inference (1.5–2×, not implemented)
6. **vDSP-ize OLA loops** — not worth it at this scale (~1.2×)

---

## Gotchas checklist

- [ ] **`aten::Int` errors** = a shape value went through Python. Use
      closure-captured ints, precomputed buffers, or `-1` wildcard.
- [ ] **`prim::NumToTensor` parents** in the graph tell you the culprit op —
      grep for them when debugging.
- [ ] **einops `rearrange` with `b=b` kwarg** always emits `aten::Int`,
      even when `b` is a Python int — use `permute`+`reshape` instead.
- [ ] **`rotary_embedding_torch`** never traces cleanly. Always bypass with
      precomputed cos/sin buffers + static `_rotate` function. Match the
      **interleaved** rotate-half convention, not the split-in-half one.
- [ ] **Match torch/torchaudio versions** exactly, or torchaudio's `.so`
      will fail to dlopen with `Symbol not found: _aoti_torch_abi_version`.
- [ ] **`torch.jit.trace` rejects dict outputs**. Return a tuple, name your
      CoreML outputs in order.
- [ ] **FP16 overflow warnings during conversion** — usually safe (CoreML
      clamps), but check outputs if something sounds wrong. Full-graph
      conv-STFT basis triggered this; spec-only and Beat This! did not.
- [ ] **First `.mlpackage` load triggers ANE compilation** (30–60 s).
      Subsequent runs use cached `.mlmodelc`. Ship the compiled version if
      startup time matters.
- [ ] **Per-chunk validation passes but full-file null test fails** → check
      edge padding, OLA window, fade-in/out logic.
- [ ] **Output 6 dB quiet** → inverse vDSP scaling is `1/N` instead of `1/(2N)`.
- [ ] **Random `N(0,1)` validation looks bad** for log-mel models → use
      input with realistic magnitude (mean ~7, std ~1).
- [ ] **ANE is slower than CPU+GPU** for transformer-heavy graphs → don't
      assume ANE wins; profile.

---

## Reference: known patches per architecture

### TFC-TDF-Net (MDX)
- ConvSTFT/ConvISTFT replacement (`F.conv1d` + `F.conv_transpose1d`)
- Static reshapes in `cac2cws` / `cws2cac` helpers
- Spec-only wrapper to externalize STFT

### BeatThis (RoFormer)
- `PartialRoformer.forward` / `PartialFTTransformer.forward` — static permute+reshape, hardcode batch=1
- `roformer.Attention.forward` — closure-captured (heads, dim_head, n, b) per module via forward-hook probe
- `_static_rotate` + precomputed cos/sin buffers replacing `rotate_queries_or_keys`
- `SumHead.forward` — tuple return + permute+index split
- `einops.layers.torch.Rearrange` in stem `nn.Sequential` → custom `nn.Module`s
