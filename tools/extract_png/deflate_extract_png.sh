#!/bin/sh
./read_png  hsf_2_00001.png
cat hsf_2_00001.inflate | ./z -d > hsf_2_00001.pix


