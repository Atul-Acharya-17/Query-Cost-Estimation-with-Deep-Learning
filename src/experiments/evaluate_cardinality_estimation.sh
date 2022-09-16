#!/bin/bash

source venv/bin/activate

HOME="/home/atul/"

dataset=$1

results="${HOME}Desktop/Query-Cost-Estimation-with-Deep-Learning/results/"

file_name="${results}${dataset}_inference.log"

exp_dir="${HOME}Desktop/Query-Cost-Estimation-with-Deep-Learning/repositories/AreCELearnedYet"
cd ${exp_dir}

naru='original-resmade_hid32,32,32,32_emb4_ep100_embedInembedOut_warm0-123'
mscn='original_base-mscn_hid16_sample1000_ep200_bs1024_100k-123'
lw_nn='original_base-lwnn_hid128_64_32_bin200_ep500_bs32_10k-123'
lw_tree='original_base-lwxgb_tr16_bin200_10k-123'
deepdb='original-spn_sample48842_rdc0.3_ms0.01-123'

echo "NARU "${naru}"" > "${file_name}"
just test-naru "${naru}" "${dataset}" >> "${file_name}" 2>&1

echo "MSCN "${mscn}"" >> "${file_name}"
just test-mscn "${mscn}" "${dataset}" >> "${file_name}" 2>&1

echo "LW-NN "${mscn}"" >> "${file_name}"
just test-lw-nn "${lw_nn}" "${dataset}" >> "${file_name}" 2>&1

echo "LW-TREE "${mscn}"" >> "${file_name}"
just test-lw-tree "${lw_tree}" "${dataset}" >> "${file_name}" 2>&1

echo "DEEPDB "${mscn}"" >> "${file_name}"
just test-deepdb "${deepdb}" "${dataset}" >> "${file_name}" 2>&1

deactivate