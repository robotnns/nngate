#!/bin/sh
g++ -g --std=c++11 -o gen_filter  gen_filter.cpp -I../libcnn -lsqlite3 ../libcnn/libcnn.so

