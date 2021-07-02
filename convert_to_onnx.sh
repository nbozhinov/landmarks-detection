#!/bin/bash
python3 -m tf2onnx.convert --saved-model trained_model --output trained_model.onnx
