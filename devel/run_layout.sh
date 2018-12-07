#!/bin/bash -l

THRESHOLD=25
N_U=100
N_OM=128
N=

THREADS=6
REALISATIONS=1
BANDWIDTH=10



# DON'T CHANGE ANYTHING ELSE.
LOG=1
SHAPE=0
NSPOKES=6
U0MIN=30

# Set a default value for the task id.
if [ ! ${SLURM_ARRAY_TASK_ID} ]; then
    SLURM_ARRAY_TASK_ID=1
fi

case "${SLURM_ARRAY_TASK_ID}" in
    [7-9])
        LOG=0
        ;;
esac


case "${SLURM_ARRAY_TASK_ID}" in
    1)
        KIND=circle
        U0MIN=30
        ;;
    2)
        KIND=filled_circle
        SHAPE=0
        ;;
    3)
        KIND=filled_circle
        SHAPE=1
        ;;
    [4-9])
        KIND=spokes
        ;;
    10)
        KIND=rlx_boundary
        ;;
    11)
        KIND=rlx_grid
        ;;
    12)
        KIND=hexagon
        ;;
    13)
        KIND=sparse
        ;;
esac

case "${SLURM_ARRAY_TASK_ID}" in
    4)
        NSPOKES=4
        U0MIN=80
        ;;
    5)
        NSPOKES=5
        U0MIN=50
        ;;
    6)
        NSPOKES=6
        U0MIN=50
        ;;
    7)
        NSPOKES=4
        ;;
    8)
        NSPOKES=5
        ;;
    9)
        NSPOKES=6
        ;;
esac

if [ $NUMERICAL == 1 ]
then
    NUMERICAL=--numerical
else
    NUMERICAL=--analytic
fi

if [ $LOG == 1 ]
then
    LOG=--log
else
    LOG=--linear
fi

if [ $VERBOSE == 1 ]
then
    VERBOSE=--verbose
else
    VERBOSE=--not-verbose
fi

jobscript="run_layout ${KIND} -N ${N} --prefix outputs/ --shape ${SHAPE} --nspokes ${NSPOKES}"
jobscript="${jobscript} --ugrid-size ${N_U} --omega-grid-size ${N_OM} --u0-min ${U0MIN} --threads ${THREADS}"
jobscript="${jobscript} ${VERBOSE} ${LOG}"
jobscript="${jobscript} --threshold ${THRESHOLD} -r ${REALISATIONS} --bw=${BANDWIDTH}"

echo "time ${jobscript}"

time ${jobscript}

