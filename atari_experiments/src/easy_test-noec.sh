#!/bin/bash
declare -a arr_s=(654 987 148 428 734 3524 4684 9128 7231 9235)
declare -a arr_ts=(21632 25246 28027 28796 44092 53826 58958 62428 74796 94039)
declare -a archs=('original' 'contract' 'contract_action_history' 'contract_dfa_state' 'contract_graph_emb')

for i in "${arr_s[@]}"
do
    for j in "${arr_ts[@]}"
    do
    	for ar in "${archs[@]}"
    	do

        t1="sbatch slurm.sh run_atari.py --task=test --env-name BreakoutDeterministic-v4 --contract dithering --train_seed $i --test_seed $j --arch $ar"
        echo $t1
        eval $t1
        t2="sbatch slurm.sh run_atari.py --task=test --env-name BreakoutDeterministic-v4 --contract actuation --train_seed $i --test_seed $j --arch $ar"
        echo $t2
        eval $t2
        t3="sbatch slurm.sh run_atari.py --task=test --env-name SpaceInvadersDeterministic-v4 --contract dithering --train_seed $i --test_seed $j --arch $ar"
        echo $t3
        eval $t3
        t4="sbatch slurm.sh run_atari.py --task=test --env-name SpaceInvadersDeterministic-v4 --contract actuation --train_seed $i --test_seed $j --arch $ar"
        echo $t4
        eval $t4
        t5="sbatch slurm.sh run_atari.py --task=test --env-name SeaquestDeterministic-v4 --contract upto4 --train_seed $i --test_seed $j --arch $ar"
        echo $t5
        eval $t5
        t6="sbatch slurm.sh run_atari.py --task=test --env-name SeaquestDeterministic-v4 --contract actuation --train_seed $i --test_seed $j --arch $ar"
        echo $t6
        eval $t6

        done
    done
done
