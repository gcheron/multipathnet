#!/bin/bash
set -e
cd /sequoia/data1/gcheron/code/torch/multipathnet/

export PYTHONPATH=/sequoia/data1/gcheron/code/torch/coco/PythonAPI

testdir=/sequoia/data2/gcheron/multipathnet_results/logs/fastrcnn_daly_app_FINAL
if [ ! -z "$INFO" ]; then
   INFOD="_$INFO"
   testdir=/sequoia/data2/gcheron/multipathnet_results/logs/fastrcnn_daly_flow_FINAL
fi

export proposals=keyframes
export test_set=testkeyframes$INFOD
export dataset=daly
export year=""

export test_best_proposals_number=10000
export test_model=$testdir/model_final.t7 
export test_save_res=$testdir

if [ ! -z "$track_setid" ];
then
   echo "RUN TEST ON TRACK ID: $track_setid"
   export test_save_res=$test_save_res/result_set_$track_setid
   export proposals=tracks
   export test_set=testtracks_set_$track_setid$INFOD
   mkdir -p $test_save_res
fi

th run_test.lua
