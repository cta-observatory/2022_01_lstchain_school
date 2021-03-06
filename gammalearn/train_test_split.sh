#!/bin/bash
# do the train/test split of DL1 data if you have not done so during the DL1 to DL2 lecture
# go to the location of this script and run ./train_test_split.sh
# you may need to modify the location of DATADIR if you have not synched it to the base of the school repository

DATADIR=../data
DL1DIR=$DATADIR/mc/DL1

for PARTDIR in `ls $DL1DIR`; do
mkdir -p $DL1DIR/$PARTDIR/testing_gl;
mkdir -p $DL1DIR/$PARTDIR/training_gl;
cp `ls $DL1DIR/$PARTDIR/*.h5 | head -n 2` $DL1DIR/$PARTDIR/testing_gl
cp `ls $DL1DIR/$PARTDIR/*.h5 | tail -n 2` $DL1DIR/$PARTDIR/training_gl;
done
