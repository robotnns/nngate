#!/bin/sh
g++ -g --std=c++11 -o sf_viewer  sf_viewer.cpp  -lsfml-graphics -lsfml-window -lsfml-system -I../libcnn ../libcnn/libcnn.so

