# AMD Ryzen AI NPU backend

This branch adds an explicit `coli npu probe` path for Windows systems with
AMD Ryzen AI Software installed.

## Why this is a bridge, not a transparent native-engine switch

Colibri's native engine is a custom C runtime for disk-streamed expert tensors.
The AMD Ryzen AI NPU runtime accepts compiled ONNX subgraphs through
`VitisAIExecutionProvider`; it does not expose a general-purpose C GEMM ABI
that can consume Colibri's arbitrary quantized tensor views one operation at a
time.

The bridge therefore provides a small, explicit ONNX execution seam first. It
never claims that the normal native `coli run` path is NPU accelerated until a
model graph can be partitioned and its numerical behavior is validated.

## Usage

Run from the `ryzen-ai-1.8.0` Conda environment, with the Ryzen AI deployment
directory available on `PATH`:

```cmd
python C:\Users\sutto\colibri-ryzen-ai-npu\c\coli npu probe
```

The command creates a tiny MatMul ONNX graph, runs it with
`VitisAIExecutionProvider` preferred over CPU, enables ONNX Runtime profiling,
and reports the selected providers and output checksum.

## Verified milestone

On the development machine, the command returned:

```text
providers: VitisAIExecutionProvider, CPUExecutionProvider
selected: VitisAIExecutionProvider
output_checksum: 4096.0
```

The same run emitted AMD AIE compiler output and a Vitis AI kernel profiling
event (`provider: VitisAIExecutionProvider`, `op_name: vitis_ai_ep_1`).

## Current limitation

The installed NPU driver is `32.0.20102.3930`. FastFlowLM documentation examined
for this project requires `32.0.203.311` or newer, so full model validation may
require a signed driver update and reboot. Provider availability and a tiny
probe are not equivalent to successful Colibri model inference.

## Next engineering stage

1. Add a model-to-ONNX subgraph adapter for a supported dense layer.
2. Define tensor layout, quantization, shape, and lifetime contracts.
3. Compare the NPU result with Colibri's CPU reference on deterministic fixtures.
4. Add explicit fallback diagnostics; never silently report CPU as NPU.
5. Only then consider routing selected Colibri layers or experts through the
   bridge during normal inference.
