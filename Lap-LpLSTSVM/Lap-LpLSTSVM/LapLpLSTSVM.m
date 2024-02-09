function [PredictY time]= LapLPLSTSVM(TestX,Data,FunPara,weight, knn,p)

tic;
A = Data.X((Data.Y==1),:);
B = Data.X((Data.Y==-1),:);
K = Data.X;
m1 = size(A,1); 
m2 = size(B,1);
m = size(Data.X,1);
n = size(Data.X,2);
c1 = FunPara.p1; 
c2 = FunPara.p2; 
c3 = c2;

e1 = ones(m1,1); 
e2=ones(m2,1); 
e = ones(m,1);
kerfPara = FunPara.kerfPara;

% L = laplacian(3,Data.X); 
L = laplacianSun2(Data.X,weight, knn);
% L = laplacian( 6,Data.X);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Cache kernel matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~strcmp(kerfPara.type,'lin')    
    K = kernelfun(Data.X,kerfPara);
    A = kernelfun(A,kerfPara,Data.X);
    B = kernelfun(B,kerfPara,Data.X);
end

H = [A,e1];  
G = [B,e2]; 
L = (L + L')/2;
[V U] = eig(L);
J = U^(1/2)*V*[Data.X,ones(size(Data.X,1),1)];
     
%J = [K,e]; 
% p = 1.5;

Z1 = zeros(size(A, 2) + 1, 1);
Z1new = ones(size(A, 2), 1);
Z1new = [Z1new' 10^-4]';
% Z1new = rand(size(A,2) + 1, 1);
% Z1new = Z1new / norm(Z1new);
iter = 1;
itmax = 20;
eps = 10^-4;

while (norm(Z1 - Z1new) >= 0.0001 && iter <= itmax)
   
    Z1 = Z1new;
    D1 = [];
    for i = 1 : size(A,1)
        D1 = [D1 1/abs(H(i,:) * Z1 + eps)^(2 - p)];
    end
    D1 = diag(D1);

    D2 = [];
    for i = 1 : size(A,2)
        D2 = [D2 1 / abs(Z1(i))^(2 - p)];
    end
    D2 = [D2 1];
    D2 = diag(D2);

    D3 = [];
    for i = 1 : size(B, 1)
        D3 = [D3 1/ abs(G(i,:) * Z1 + 1 + eps)^(2-p)];
    end
    D3 = diag(D3);

    D4 = [];
    for i = 1 : size(K, 1)
        D4 = [D4 1 / abs(J(i, :) * Z1 + eps)^(2-p)];
    end
    D4 = diag(D4);

    matr = H'*D1*H + c1*D2 + c2*G'*D3*G + c3*J'*D4 * J;

    Z1new = matr \(c2 * G' * D3 * e2);      
   
    iter = iter + 1;
      
end


Z2 = zeros(size(A, 2) + 1, 1);
Z2new = ones(size(A, 2), 1);
Z2new = [Z2new' 10^-4]';
% Z2new = rand(size(A,2) + 1, 1);
% Z2new = Z2new / norm(Z2new);
iter = 1;
while(norm(Z2 - Z2new) > 0.0001 && iter <= itmax)
    
    Z2 = Z2new;
    D1 = [];
    for i = 1 : size(B,1)
        D1 = [D1 1/abs(G(i,:) * Z2 + eps)^(2-p)];
    end
    D1 = diag(D1);
    
    D2 = [];
    for i = 1 : size(Z2, 1)
        D2 = [D2 1 / abs(Z2(i))^(2-p)];
    end
    D2 = diag(D2);
    
    D3 = [];
    for i = 1 : size(A, 1)
        D3 = [D3 1 / abs(-H(i,:) * Z2 + 1 + eps)^(2-p)];
    end
    D3 = diag(D3);
    
    D4 = [];
    for i = 1 : size(K, 1)
        D4 = [D4 1 / abs(J(i,:) * Z2 + eps)^(2-p)];
    end
    D4 = diag(D4);
    mat=G' * D1 * G + c1*D2 + c2*H'*D3*H + c3*J'*D4 * J;

%     mat=(mat+mat')/2;
    Z2new = mat\(c2 * H' * D3 * e1);
    
    iter = iter + 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train classifier using Eig solver
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m3 = size(TestX,1);   %���ֳ����Ĳ��Լ�
e = ones(m3,1);

K = [TestX, e];
v1 = Z1;
v2 = Z2;
time = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Predict 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if ~strcmp(kerfPara.type,'lin')    
    w1 = sqrt(v1(1:m)'*K*v1(1:m));
    w2 = sqrt(v2(1:m)'*K*v2(1:m));
    K = [kernelfun(TestX,kerfPara,Data.X),e];
else
	w1 = sqrt(v1(1:n)'*v1(1:n));
    w2 = sqrt(v2(1:n)'*v2(1:n));
    K = [TestX, e];    
end

PredictY = sign(abs(K*v2/w2)-abs(K*v1/w1));

