function [accura,time] = LapLPLSTSVMTest(X, label, weight, knn, p)

A = X(find(label==1),:);
B = X(find(label==-1),:);
view=[A;B];

n = size(view,1);
h1 = round(0.08 * n);
h2 = round(0.6 * n);

add = 10233;

nn = 5;
NN = size(X,1);

for iter = 1:nn
    iter;
    t1=clock;
    seed =2^iter+add; 
    randn('seed',seed),rand('seed',seed);
    S = randperm(NN);
    ltrain = S(1:h1);     
    utrain = S(h1+1:h2); 
    test = S(h2+1:end);  
    Xtr.data = view(ltrain,:);   
    Xtr.L = label(ltrain,:);  
    Xtr.u = view(utrain,:);  
    Xte.data = view(test,:); 
    Xte.L = label(test,:);
    
    
    [nu,mu] = funLapLpLSTSVM(Xtr, weight, knn, p);
    FunPara.p1=nu;
    FunPara.p2=mu;
    FunPara.kerfPara.type = 'lin';
    Data.X=[Xtr.data;Xtr.u];   
    Data.Y=[Xtr.L;zeros(size(Xtr.u,1),1)]; 
    [pre, time] = LapLpLSTSVM(Xte.data,Data,FunPara,weight, knn, p);
 
    accura(iter,:)=length(find(abs(pre-Xte.L)<1e-5))/length(pre);
    
end
    accura = accura';
    temp = accura; 
    accura = [accura mean(temp)];
    accura = [accura std(temp)];
end
