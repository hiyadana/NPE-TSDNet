%% Coarseness���ֲڶȡ���by:������
function IR = coarseness( graypic)%graphicΪ������ĻҶ�ͼ��
h_m=[1 -1];%ˮƽ��ģ
v_m=h_m';%��ֱ��ģ
graypic=double(graypic);
%ˮƽ�������ز����ֵ�ۼ�
I_h = conv2(graypic, h_m);
I_h(:,1)=[];
I_h(:,end)=[];
I_h_L1 = sum(sum(abs(I_h)));
%I_h_L1 = double(I_h_L1);
%��ֱ�������ز����ֵ�ۼ�
I_v=conv2(graypic,v_m);
I_v(1,:)=[];
I_v(end,:)=[];
I_v_L1 = sum(sum(abs(I_v)));
%I_v_L1 = double(I_v_L1);
%ͼ������ֵ�ۼ�
I_L1 = sum(sum(abs(graypic)));
%I_L1 = double(I_L1);
%�ֲڶ� 
IR=(I_h_L1 + I_v_L1)/I_L1;
end 