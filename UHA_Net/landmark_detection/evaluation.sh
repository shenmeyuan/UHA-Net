#!/bin/bash
model=drunetpp
runname=drunetpp_DSC
phase=test
result=034

resultp1=$(printf "%03d" `expr $result + 1`)
FILE=../runs/$runname/results/${phase}_epoch${resultp1}
if [ ! -e "$FILE" ]; then
    checkpoint=$(ls ../runs/$runname/checkpoints/*${runname}_epoch${result}*.pt)
    python main.py -d ../runs -r $runname -p test -m $model -c $checkpoint
fi
python evaluation.py -i ../runs/$runname/results/${phase}_epoch${resultp1}/ > ../runs/$runname/results/${phase}_epoch${resultp1}/eval.txt
