#!/bin/bash

START_NUM=1
END_NUM=25
LOG_SUFFIX=`date +"%Y_%m_%d_%H_%M_%S"`

for i in `seq $START_NUM $END_NUM`
do
  python manage.py --key-pair=next_key_1 --identity-file=/Users/kevinjamieson/aws_keys/next_key_1.pem --instance-type=c3.large launch kevin_research_random_"$i" > /tmp/cluster-"$i"-"$LOG_SUFFIX".out 2>&1 &
  sleep 15
done
