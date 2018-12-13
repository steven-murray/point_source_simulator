#!/usr/bin/env bash

# This runs all combinations of parameters for fiducial investigations for the Ray Bolger project.

PREFIX=outputs
UMIN=10
UMAX=150
UGRID_SIZE=40
OMEGA_GRID_SIZE=32
U0MIN=${UMIN}

THREADS=4
PROCESSES=1
TAPER=blackman
THRESHOLD=10
DIAMETER=1.5

SKY_MOMENT=2
SMAX=1000

REALISATIONS=1

BANDWIDTH=10 # MHz

# these will be changed for the layouts where needed.
SHAPE=0
LOG=0
NSPOKES=1
BASE=3
LARGE=0

for N_ANTENNA in 10000 # 256 1000 #10000
do
    for LAYOUT in 1 2 3 4 5
    do
        case "${LAYOUT}" in
            1)
                KIND=spokes
                LOG=0
                NSPOKES=3
                BASE=3
                ;;
            2)
                KIND=spokes
                LOG=1
                NSPOKES=4
                BASE=3
                ;;
            3)
                KIND=filled_circle
                SHAPE=0
                ;;
            4)
                KIND=filled_circle
                SHAPE=1
                ;;
            5)
                KIND=spokes
                LOG=1
                NSPOKES=2
                BASE=3
                LARGE=1
        esac

        if [ $LOG == 1 ]
        then
            LOG=--log
        else
            LOG=--linear
        fi

        if [ $LARGE == 1 ]
        then
            LARGE=--large
        else
            LARGE=--not-large
        fi

        jobscript="run_layout ${KIND} -N ${N_ANTENNA} -p ${PREFIX} ${LOG} --shape ${SHAPE} -n ${NSPOKES} -u ${UMIN}"
        jobscript="${jobscript} -U ${UMAX} --ugrid-size ${UGRID_SIZE} --omega-grid-size ${OMEGA_GRID_SIZE} -m ${U0MIN}"
        jobscript="${jobscript} -t ${THREADS} --taper ${TAPER} -h ${THRESHOLD} -d ${DIAMETER} -r ${REALISATIONS} --bw ${BANDWIDTH}"
        jobscript="${jobscript} -j ${PROCESSES} --sky-moment ${SKY_MOMENT} --smax ${SMAX} ${LARGE}"

        echo "time ${jobscript} --restart"

        time ${jobscript} --restart
    done
done
