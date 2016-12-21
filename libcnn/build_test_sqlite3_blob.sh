#!/bin/sh
g++ -g --std=c++11 -o test_sqlite3_blob   test_sqlite3_blob.cpp  -lsqlite3 ./libcnn.so

