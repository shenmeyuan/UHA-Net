#!/bin/bash
runname=zy
model=gln
times=50
#rm -rf ../runs/$runname
nohup python3 main.py -d ../runs -r $runname -p train -m $model -l u2net -e $times -n cephalometric > $model.log &
