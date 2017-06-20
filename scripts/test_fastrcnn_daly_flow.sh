#!/bin/bash
cd /sequoia/data1/gcheron/code/torch/multipathnet/

export PYTHONPATH=/sequoia/data1/gcheron/code/torch/coco/PythonAPI

export proposals=keyframes
export test_set=testkeyframes_flow
export dataset=daly
export year=""

export test_best_proposals_number=10000
export test_model=/sequoia/data2/gcheron/mutlipathnet_results/logs/fastrcnn_daly_flow_FINAL/model_final.t7 
export test_save_res=/sequoia/data2/gcheron/mutlipathnet_results/logs/fastrcnn_daly_flow_FINAL

th run_test.lua
