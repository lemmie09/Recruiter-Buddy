//Scilab Code for finding Coprime Factorization 
//Written By Shalini Shrivastava
mode()
clc()
s=%s;
num=input("Enter the numerator of the Transfer function");
denum=input("Enter the denominator of the transfer function");
Zin=syslin('c',num/denum);// Transfer function Defination
tfss=tf2ss(Zin);// Transfer function to state space
tfss=minss(tfss,0.01); // Minimal Realization 
A=tfss(2);B=tfss(3);C=tfss(4);D=tfss(5);//State Space A, B, C, D
p=input("Pole placement"); // Enter the pole in the matrix form [ -1 -2 ]
Fp=ppol(A,B,p);F=-Fp;Hp=ppol(A,C',p);H=-Hp;
i=size(A);i=i(1);
Ds=1+F*(s*eye(i,i)-(A+B*F))^(-1)*B;
Ns=(C+D*F)*(s*eye(i,i)-(A+B*F))^(-1)*B+D;
Xs=F*(s*eye(i,i)-(A+H'*C))^(-1)*H';
Ys=1+F*(s*eye(i,i)-(A+H'*C))^(-1)*(-B-H'*D);

//Checking Bezout's Identity
BZ=Ns*Xs+Ys*Ds;n=(roots(numer(BZ)));d=(roots(denom(BZ)));        
if n=d
    disp{"Bezouts Identity Hold, Hurray we can proceed furture"}
end

//Calculation of Zc& Zt
Q=input("Enter the Q"); 
ZC=(Ys-Ns*Q)/(Xs+Ds*Q); // Controller Transfer Fuction
Zminreal=ss2tf(tfss);   //Minimization of Orignal Trafnfer funtion
Zcs=tf2ss(ZC);
ZcsMin=minss(Zcs,0.01);
ZcMin=ss2tf(ZcsMin);
TF=Ns*[Ys-Ns*Q];//Total Transfer Function

//BODE PLOT 
fmin=input("Minimium frequency for bode plot");
fmax=input("Minimium frequency for bode plot");
TF=syslin('c',TF); //Linear system definition of Total Tranfer Function
ZC=syslin('c',ZC); //Linear system definition of Compensator Tranfer Function
Zin=syslin('c',Zin); //Linear system definition of Given system Tranfer Function
clf();
bode([tf;ZC;TF],fmin,fmax,['Zin';'ZC';'ZT']);