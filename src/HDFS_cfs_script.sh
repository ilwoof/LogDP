#!/bin/bash

cp -R ../CCD ./CCD 
cd ./CCD 
rm mb/mb.out
rm indicator1/indicator.out
echo start focused variable 1/17 in label:0----------
./main ../../data/HDFS_session/raw/HDFS_session_training_data.txt "" 0.05 IAMB 0 "" -1
sleep 3s
echo start focused variable 2/17 in label:0----------
./main ../../data/HDFS_session/raw/HDFS_session_training_data.txt "" 0.05 IAMB 1 "" -1
sleep 3s
echo start focused variable 3/17 in label:0----------
./main ../../data/HDFS_session/raw/HDFS_session_training_data.txt "" 0.05 IAMB 2 "" -1
sleep 3s
echo start focused variable 4/17 in label:0----------
./main ../../data/HDFS_session/raw/HDFS_session_training_data.txt "" 0.05 IAMB 3 "" -1
sleep 3s
echo start focused variable 5/17 in label:0----------
./main ../../data/HDFS_session/raw/HDFS_session_training_data.txt "" 0.05 IAMB 4 "" -1
sleep 3s
echo start focused variable 6/17 in label:0----------
./main ../../data/HDFS_session/raw/HDFS_session_training_data.txt "" 0.05 IAMB 5 "" -1
sleep 3s
echo start focused variable 7/17 in label:0----------
./main ../../data/HDFS_session/raw/HDFS_session_training_data.txt "" 0.05 IAMB 6 "" -1
sleep 3s
echo start focused variable 8/17 in label:0----------
./main ../../data/HDFS_session/raw/HDFS_session_training_data.txt "" 0.05 IAMB 7 "" -1
sleep 3s
echo start focused variable 9/17 in label:0----------
./main ../../data/HDFS_session/raw/HDFS_session_training_data.txt "" 0.05 IAMB 8 "" -1
sleep 3s
echo start focused variable 10/17 in label:0----------
./main ../../data/HDFS_session/raw/HDFS_session_training_data.txt "" 0.05 IAMB 9 "" -1
sleep 3s
echo start focused variable 11/17 in label:0----------
./main ../../data/HDFS_session/raw/HDFS_session_training_data.txt "" 0.05 IAMB 10 "" -1
sleep 3s
echo start focused variable 12/17 in label:0----------
./main ../../data/HDFS_session/raw/HDFS_session_training_data.txt "" 0.05 IAMB 11 "" -1
sleep 3s
echo start focused variable 13/17 in label:0----------
./main ../../data/HDFS_session/raw/HDFS_session_training_data.txt "" 0.05 IAMB 12 "" -1
sleep 3s
echo start focused variable 14/17 in label:0----------
./main ../../data/HDFS_session/raw/HDFS_session_training_data.txt "" 0.05 IAMB 13 "" -1
sleep 3s
echo start focused variable 15/17 in label:0----------
./main ../../data/HDFS_session/raw/HDFS_session_training_data.txt "" 0.05 IAMB 14 "" -1
sleep 3s
echo start focused variable 16/17 in label:0----------
./main ../../data/HDFS_session/raw/HDFS_session_training_data.txt "" 0.05 IAMB 15 "" -1
sleep 3s
echo start focused variable 17/17 in label:0----------
./main ../../data/HDFS_session/raw/HDFS_session_training_data.txt "" 0.05 IAMB 16 "" -1
sleep 3s
mv mb/mb.out ../../data/HDFS_session/raw/HDFS_session_training_data_mb.txt
mv indicator1/indicator.out ../../data/HDFS_session/raw/HDFS_session_training_data_tm.txt
cd ..
rm -rf ./CCD 
echo label 0 done.
