#!/bin/bash

START_NUM=1
END_NUM=5
LOG_SUFFIX=`date +"%Y_%m_%d_%H_%M_%S"`

# for i in `seq $START_NUM $END_NUM`
# do
#   export WORKER_WORKING_DIR="/generic_worker/random_full"
#   export WORKER_COMMAND="python ./main.py"
#   machine_name=kevin_research_random_full_"$i"
#   log_name=cluster-random-"$i"-"$LOG_SUFFIX".out
#   echo "$machine_name"-"$log_name"
#   python manage.py --key-pair=next_key_1 --identity-file=/Users/kevinjamieson/aws_keys/next_key_1.pem --instance-type=c1.medium destroy "$machine_name" > /tmp/"$log_name" 2>&1 &
#   sleep 15
# done

# for i in `seq $START_NUM $END_NUM`
# do
#   export WORKER_WORKING_DIR="/generic_worker/nvb_batch"
#   export WORKER_COMMAND="python ./main.py"
#   machine_name=kevin_research_nvb_batch_"$i"
#   log_name=cluster-batch-"$i"-"$LOG_SUFFIX".out
#   echo "$machine_name"-"$log_name"
#   python manage.py --key-pair=next_key_1 --identity-file=/Users/kevinjamieson/aws_keys/next_key_1.pem --instance-type=c1.medium destroy "$machine_name" > /tmp/"$log_name" 2>&1 &
#   sleep 15
# done

# for i in `seq $START_NUM $END_NUM`
# do
#   export WORKER_WORKING_DIR="/generic_worker/nvb_iter"
#   export WORKER_COMMAND="python ./main.py"
#   machine_name=kevin_research_nvb_iter_"$i"
#   log_name=cluster-iter-"$i"-"$LOG_SUFFIX".out
#   echo "$machine_name"-"$log_name"
#   python manage.py --key-pair=next_key_1 --identity-file=/Users/kevinjamieson/aws_keys/next_key_1.pem --instance-type=c3.large destroy "$machine_name" > /tmp/"$log_name" 2>&1 &
#   sleep 15
# done

# for i in `seq $START_NUM $END_NUM`
# do
#   export WORKER_WORKING_DIR="/generic_worker/bandit_batch"
#   export WORKER_COMMAND="python ./main.py"
#   machine_name=kevin_research_bandit_batch_"$i"
#   log_name=cluster-bandit_batch-"$i"-"$LOG_SUFFIX".out
#   echo "$machine_name"-"$log_name"
#   python manage.py --key-pair=next_key_1 --identity-file=/Users/kevinjamieson/aws_keys/next_key_1.pem --instance-type=c3.large destroy "$machine_name" > /tmp/"$log_name" 2>&1 &
#   sleep 15
# done

# for i in `seq $START_NUM $END_NUM`
# do
#   export WORKER_WORKING_DIR="/generic_worker/bandit_iter"
#   export WORKER_COMMAND="python ./main.py"
#   machine_name=kevin_research_bandit_iter_"$i"
#   log_name=cluster-bandit_iter-"$i"-"$LOG_SUFFIX".out
#   echo "$machine_name"-"$log_name"
#   python manage.py --key-pair=next_key_1 --identity-file=/Users/kevinjamieson/aws_keys/next_key_1.pem --instance-type=c3.large destroy "$machine_name" > /tmp/"$log_name" 2>&1 &
#   sleep 15
# done


# for i in `seq $START_NUM $END_NUM`
# do
#   export WORKER_WORKING_DIR="/generic_worker/rr_batch"
#   export WORKER_COMMAND="python ./main.py"
#   machine_name=kevin_research_rr_batch_"$i"
#   log_name=cluster-rr_batch-"$i"-"$LOG_SUFFIX".out
#   echo "$machine_name"-"$log_name"
#   python manage.py --key-pair=next_key_1 --identity-file=/Users/kevinjamieson/aws_keys/next_key_1.pem --instance-type=c3.large destroy "$machine_name" > /tmp/"$log_name" 2>&1 &
#   sleep 15
# done


# for i in `seq $START_NUM $END_NUM`
# do
#   export WORKER_WORKING_DIR="/generic_worker/rr_iter"
#   export WORKER_COMMAND="python ./main.py"
#   machine_name=kevin_research_rr_iter_"$i"
#   log_name=cluster-rr_iter-"$i"-"$LOG_SUFFIX".out
#   echo "$machine_name"-"$log_name"
#   python manage.py --key-pair=next_key_1 --identity-file=/Users/kevinjamieson/aws_keys/next_key_1.pem --instance-type=c3.large destroy "$machine_name" > /tmp/"$log_name" 2>&1 &
#   sleep 15
# done

for i in `seq $START_NUM $END_NUM`
do
  export WORKER_WORKING_DIR="/generic_worker/hyperband_iter"
  export WORKER_COMMAND="python ./main.py"
  machine_name=kevin_research_hyperband_iter_"$i"
  log_name=cluster-hyperband_batch-"$i"-"$LOG_SUFFIX".out
  echo "$machine_name"-"$log_name"
  python manage.py --key-pair=next_key_1 --identity-file=/Users/kevinjamieson/aws_keys/next_key_1.pem --instance-type=c1.medium launch "$machine_name" > /tmp/"$log_name" 2>&1 &
  sleep 15
done

for i in `seq $START_NUM $END_NUM`
do
  export WORKER_WORKING_DIR="/generic_worker/random_full"
  export WORKER_COMMAND="python ./main.py"
  machine_name=kevin_research_random_full_"$i"
  log_name=cluster-hyperband_batch-"$i"-"$LOG_SUFFIX".out
  echo "$machine_name"-"$log_name"
  python manage.py --key-pair=next_key_1 --identity-file=/Users/kevinjamieson/aws_keys/next_key_1.pem --instance-type=c1.medium launch "$machine_name" > /tmp/"$log_name" 2>&1 &
  sleep 15
done