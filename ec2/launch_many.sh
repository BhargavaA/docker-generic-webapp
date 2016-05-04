#!/bin/bash

START_NUM=1
END_NUM=1
LOG_SUFFIX=`date +"%Y_%m_%d_%H_%M_%S"`

for i in `seq $START_NUM $END_NUM`
do
	export WORKER_WORKING_DIR="/generic_worker/simcode"
	export WORKER_COMMAND="python main.py"
	machine_name=triplets_research_"$i"
	log_name="$machine_name"-"$LOG_SUFFIX".out
	echo "$machine_name"-"$log_name"
	python manage.py --key-pair=next_key_1 --identity-file=/Users/kevinjamieson/aws_keys/next_key_1.pem --instance-type=c4.8xlarge --spot-price=1.675 launch "$machine_name" > /tmp/"$log_name" 2>&1 &
	sleep 15
done