clear all;
close all;
clc;

load('./dataset/handwritten_1_4.mat');
weight = 1;
knn = 8;
data = mapminmax(data',0,1)';
p_set = [1,1.5,2,3,5];

[accuracy,time] = LapLpLSTSVMTest(data, label, weight, knn, 5);


