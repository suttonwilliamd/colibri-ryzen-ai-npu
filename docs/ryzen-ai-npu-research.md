# Ryzen AI NPU research notes

## Conclusions

The supported Windows path for AMD Ryzen AI NPU execution is:

```text
ONNX graph -> ONNX Runtime -> VitisAIExecutionProvider -> Ryzen XDNA NPU
```

Windows ML can dynamically register hardware execution providers and prefer an
NPU, but it still operates on ONNX graphs and `OrtSession::Run`. DirectML is a
legacy DirectX 12 GPU path and is not the correct proof of Ryzen NPU execution.

There is no documented public AMD API for submitting an arbitrary Colibri
quantized tensor or one standalone GEMM directly to the NPU. A Colibri adapter
must build/cache ONNX subgraphs and accept graph partitioning constraints.

## Colibri seams inspected

- Native model input: Hugging Face-style safetensors shards and optional `.qs`
  quantized sidecars (`c/st.h`, `c/colibri.c`, `docs/FORMATS.md`).
- Native tensor dispatch: `matmul_qt_ex` in `c/colibri.c`.
- Useful first target: a fixed-shape f32 dense matmul or the full dense MLP
  (`gate_proj -> up_proj -> SiLU product -> down_proj`).
- Hard later targets: dynamic MoE routing, MLA attention, RoPE, softmax, and
  persistent KV-cache ownership.
- Existing backend precedent: Vulkan/CUDA/HIP use opaque persistent handles,
  lazy upload, and return-to-CPU fallback. The NPU adapter follows that policy.

## Runtime evidence on this machine

The `ryzen-ai-1.8.0` environment reports:

```text
onnxruntime 1.27.0
VitisAIExecutionProvider
DmlExecutionProvider
CPUExecutionProvider
```

The repository command `coli npu probe` builds a deterministic MatMul ONNX
model, requests Vitis AI before CPU, enables profiling, and requires a
provider-tagged Vitis AI kernel event. It has passed with:

```text
selected: VitisAIExecutionProvider
vitis_kernel_events: 1
output_checksum: 4096.0
```

The AMD runtime also emitted AIE compiler output during the test.

## Official references

- AMD Ryzen AI documentation: https://ryzenai.docs.amd.com/en/latest/
- AMD installation and driver requirements: https://ryzenai.docs.amd.com/en/latest/inst.html
- AMD model deployment: https://ryzenai.docs.amd.com/en/latest/modelrun.html
- AMD supported operators: https://ryzenai.docs.amd.com/en/latest/ops_support.html
- AMD RyzenAI Windows ML examples: https://github.com/amd/RyzenAI-SW/tree/main/WinML
- ONNX Runtime Vitis AI EP: https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html
- Windows ML overview: https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview
- Windows ML supported EPs: https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers
- ONNX Runtime Vitis AI factory ABI: https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/vitisai/vitisai_provider_factory.cc

## Driver note

The locally reported NPU driver is `32.0.20102.3930`. AMD Ryzen AI 1.8
installation documentation identifies `32.0.203.280` or newer as the minimum
for the documented software line. The tiny graph probe works with the current
installation, but a full model should be revalidated after any driver update.
