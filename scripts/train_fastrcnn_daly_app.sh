#!/bin/bash

export PYTHONPATH=/sequoia/data1/gcheron/code/torch/coco/PythonAPI

export proposals=keyframes
export train_set=trainkeyframes
export test_set=testkeyframes
export dataset=daly
export year=""

export model=vgg

export nDonkeys=6


export test_best_proposals_number=2000
export best_proposals_number=2000
export test_nsamples=1000

# DEBUG
#export test_nsamples=10
#export epochSize=2
#export snapshot=3
#export nEpochs=3
#export nDonkeys=1

export save_folder=/sequoia/data2/gcheron/mutlipathnet_results/logs/fastrcnn_daly_app_${RANDOM}${RANDOM}
mkdir -p $save_folder

th train.lua | tee $save_folder/log.txt
