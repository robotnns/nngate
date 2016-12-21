#!/bin/sh
g++ -g --std=c++11 -o print_filter  print_filter.cpp -I../libcnn -lsqlite3 ../libcnn/libcnn.so

