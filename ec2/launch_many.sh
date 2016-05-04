#!/bin/bash

START_NUM=4
END_NUM=4
LOG_SUFFIX=`date +"%Y_%m_%d_%H_%M_%S"`

# for i in `seq $START_NUM $END_NUM`
# do
#   export WORKER_WORKING_DIR="/generic_worker/hyperband_iter"
#   export WORKER_COMMAND="python ./main.py"
#   machine_name=kevin_research_hyperband_iter_"$i"
#   log_name=cluster-hyperband_iter-"$i"-"$LOG_SUFFIX".out
#   echo "$machine_name"-"$log_name"
#   python manage.py --key-pair=next_key_1 --identity-file=/Users/kevinjamieson/aws_keys/next_key_1.pem --instance-type=c4.large destroy "$machine_name" > /tmp/"$log_name" 2>&1 &
#   sleep 15
# done

# for i in `seq $START_NUM $END_NUM`
# do
#   export WORKER_WORKING_DIR="/generic_worker/random_full"
#   export WORKER_COMMAND="python ./main.py"
#   machine_name=kevin_research_random_full_"$i"
#   log_name=cluster-random_full-"$i"-"$LOG_SUFFIX".out
#   echo "$machine_name"-"$log_name"
#   python manage.py --key-pair=next_key_1 --identity-file=/Users/kevinjamieson/aws_keys/next_key_1.pem --instance-type=c4.large destroy "$machine_name" > /tmp/"$log_name" 2>&1 &
#   sleep 15
# done

for i in `seq $START_NUM $END_NUM`
do
	export WORKER_WORKING_DIR="/generic_worker/weighted_solvers/python"
	export WORKER_COMMAND="python hyperband_main.py"
	export OMP_NUM_THREADS=36
  machine_name=kevin_research_kernel_"$i"
  log_name=cluster-kernel-"$i"-"$LOG_SUFFIX".out
  echo "$machine_name"-"$log_name"
  python manage.py --key-pair=next_key_1 --identity-file=/Users/kevinjamieson/aws_keys/next_key_1.pem --instance-type=c4.8xlarge --spot-price=1.675 launch "$machine_name" > /tmp/"$log_name" 2>&1 &
  sleep 15
done