%% Coarseness，粗糙度——by:王特亮
function IR = coarseness( graypic)%graphic为待处理的灰度图像
h_m=[1 -1];%水平掩模
v_m=h_m';%垂直掩模
graypic=double(graypic);
%水平方向像素差绝对值累加
I_h = conv2(graypic, h_m);
I_h(:,1)=[];
I_h(:,end)=[];
I_h_L1 = sum(sum(abs(I_h)));
%I_h_L1 = double(I_h_L1);
%垂直方向像素差绝对值累加
I_v=conv2(graypic,v_m);
I_v(1,:)=[];
I_v(end,:)=[];
I_v_L1 = sum(sum(abs(I_v)));
%I_v_L1 = double(I_v_L1);
%图像像素值累加
I_L1 = sum(sum(abs(graypic)));
%I_L1 = double(I_L1);
%粗糙度 
IR=(I_h_L1 + I_v_L1)/I_L1;
end 