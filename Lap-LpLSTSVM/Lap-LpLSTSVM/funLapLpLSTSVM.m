function [nu,mu]=funLapLPLSTSVM(Xtr,weight, knn, p)
    N1=size(Xtr.data,1);
    N2=size(Xtr.u,1);
    V=5;
    idx=crossvalind('Kfold',ones(N1,1),V);
    idy=crossvalind('Kfold',ones(N2,1),V);
    a=5;
    base=2;
    ac_v1=zeros(2*a+1,2*a+1,V);
    for m=1:V
        ltest=(idx==m);
        ltrain=~ltest;
        utest=(idy==m);
        utrain=~utest;
        XV.data=Xtr.data(ltrain,:);
        XV.u=Xtr.u(utrain,:);
        Xte.data=Xtr.data(ltest,:);
        XV.L=Xtr.L(ltrain,:);
        Xte.L=Xtr.L(ltest,:);

        FunPara.kerfPara.type = 'lin';
        Data.X=[XV.data;XV.u];
        Data.Y=[XV.L;zeros(size(XV.u,1),1)];
        for i=-a:a
            nu=base^i;
            for j=-a:a
              mu=base^j;     
                 FunPara.p1=nu;
                 FunPara.p2=mu;
              pre=LapLpLSTSVM(Xte.data,Data,FunPara,weight, knn, p);
              ac_v1(i+a+1,j+a+1,m)=length(find(abs(pre-Xte.L)<1e-5))/length(pre);

            end
        end
    end

    ac_v = sum(ac_v1,3);
    [yy0,yyi0] = sort(ac_v(:));
    yyi0 = min(find(ac_v(:) > yy0(end)-1e-6));    
    [indi,indj]=ind2sub(size(ac_v),yyi0);
    nu=base^(indi-a-1);
    mu=base^(indj-a-1);
end