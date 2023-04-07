#!/bin/bash

for (( photons=32768; photons<=4194304; photons*=2 )); do
    sed -i "s/#define PHOTONS.*/#define PHOTONS $photons/" params.h
    make clean &&  make && perf stat  ./tiny_mc
done
