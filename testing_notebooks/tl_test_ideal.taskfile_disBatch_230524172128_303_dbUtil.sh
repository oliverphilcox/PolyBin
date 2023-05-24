#!/bin/bash

export DISBATCH_KVSSTCP_HOST=10.128.146.7:33671 PYTHONPATH=/mnt/home/carriero/projects/disBatch/beta/disBatch:${PYTHONPATH}

if [[ $1 == '--mon' ]]
then
    exec /mnt/home/ophilcox/mpivenv2/bin/python3 /mnt/home/carriero/projects/disBatch/beta/disBatch/disbatch/dbMon.py /mnt/home/ophilcox/PolyBin/testing_notebooks/tl_test_ideal.taskfile_disBatch_230524172128_303
elif [[ $1 == '--engine' ]]
then
    exec /mnt/home/ophilcox/mpivenv2/bin/python3 -c 'from disbatch import disBatch ; disBatch.main()' "$@"
else
    exec /mnt/home/ophilcox/mpivenv2/bin/python3 -c 'from disbatch import disBatch ; disBatch.main()' --context /mnt/home/ophilcox/PolyBin/testing_notebooks/tl_test_ideal.taskfile_disBatch_230524172128_303_dbUtil.sh "$@" < /dev/null 1> /mnt/home/ophilcox/PolyBin/testing_notebooks/tl_test_ideal.taskfile_disBatch_230524172128_303_${BASHPID}_context_launch.log
fi
