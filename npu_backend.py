"""AMD Ryzen AI NPU backend bridge for Colibri.

Uses the installed ONNX Runtime VitisAIExecutionProvider. This is deliberately
separate from Colibri\'s native tensor engine: Ryzen AI accepts compiled ONNX
subgraphs, not arbitrary streamed C tensors.
"""
from __future__ import annotations
import argparse, json, os, sys, tempfile
from pathlib import Path

def _ort():
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise SystemExit("Ryzen AI Python environment is required: install onnxruntime-vitisai") from e
    return ort

def probe():
    import numpy as np
    import onnx
    from onnx import helper
    ort=_ort()
    x=helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1,64])
    w=helper.make_tensor("W", onnx.TensorProto.FLOAT, [64,64], np.eye(64,dtype=np.float32).ravel())
    y=helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1,64])
    graph=helper.make_graph([helper.make_node("MatMul",["X","W"],["Y"])],"colibri_ryzen_npu_probe",[x],[y],[w])
    model=helper.make_model(graph,opset_imports=[helper.make_operatorsetid("",13)]); model.ir_version=9
    with tempfile.TemporaryDirectory() as d:
        model_path=Path(d)/"probe.onnx"; onnx.save(model,str(model_path))
        so=ort.SessionOptions(); so.enable_profiling=True
        s=ort.InferenceSession(str(model_path),sess_options=so,providers=["VitisAIExecutionProvider","CPUExecutionProvider"])
        out=s.run(None,{"X":np.ones((1,64),np.float32)})[0]
        profile=s.end_profiling()
        print(json.dumps({"providers":s.get_providers(),"selected":"VitisAIExecutionProvider" if "VitisAIExecutionProvider" in s.get_providers() else None,"output_checksum":float(out.sum()),"profile":profile},indent=2))

def main(argv=None):
    ap=argparse.ArgumentParser(prog="coli npu",description="AMD Ryzen AI NPU backend")
    sub=ap.add_subparsers(dest="cmd",required=True)
    sub.add_parser("probe",help="execute a real MatMul through VitisAIExecutionProvider")
    a=ap.parse_args(argv)
    if a.cmd=="probe": probe()
if __name__=="__main__": main()
