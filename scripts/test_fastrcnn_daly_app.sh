#!/bin/bash
cd /sequoia/data1/gcheron/code/torch/multipathnet/

export PYTHONPATH=/sequoia/data1/gcheron/code/torch/coco/PythonAPI

export proposals=keyframes
export test_set=testkeyframes
export dataset=daly
export year=""

export test_best_proposals_number=2000
export test_model=/sequoia/data2/gcheron/mutlipathnet_results/logs/fastrcnn_daly_app_run2/model_final.t7 
export test_save_res=/sequoia/data2/gcheron/mutlipathnet_results/logs/fastrcnn_daly_app_run2/

th run_test.lua
