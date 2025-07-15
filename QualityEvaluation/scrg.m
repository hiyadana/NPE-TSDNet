%% scrg:信杂比增益——by:王特亮
function [s, sg]=scrg(img_in, img_out, pw, lur, luc, rdr, rdc)
%img_in是待处理图像
%img_out是处理后图像
%pw是PaddingWidth
%lur是leftupPointRow=input('请输入目标的左上角的行数(第二坐标):');
%luc是leftupPointCol=input('请输入目标的左上角的列数(第一坐标):');
%rdr是rightdownPointRow=input('请输入目标的右下角的行数(第二坐标):');
%rdc是rightdownPointCol=input('请输入目标的右下角的列数(第一坐标):');
%(row,column)

img_in_target=img_in(lur:rdr,luc:rdc);
img_out_target=img_out(lur:rdr,luc:rdc);

% lurfa = lur-pw:
% if lurfa<1
%     lurfa=1;
% end
% lursa = lur-1;
% if lursa<1
%     lursa=1;
% end
% lucfa = luc-pw;
% if lucfa<1
%     lucfa=1;
% end
% lucsa = rdc;
% 
% lurfb = lurfa;
% lursb = rdr;

img_in_padding_a = img_in(lur-pw:lur-1, luc-pw:rdc);
img_in_padding_b = img_in(lur-pw:rdr, rdc+1:rdc+pw);
img_in_padding_c = img_in(rdr+1:rdr+pw, luc:rdc+pw);
img_in_padding_d = img_in(lur:rdr+pw, luc-pw:luc-1);
img_in_padding = [img_in_padding_a(:);img_in_padding_b(:);img_in_padding_c(:);img_in_padding_d(:)];

img_out_padding_a = img_out(lur-pw:lur-1, luc-pw:rdc);
img_out_padding_b = img_out(lur-pw:rdr, rdc+1:rdc+pw);
img_out_padding_c = img_out(rdr+1:rdr+pw, luc:rdc+pw);
img_out_padding_d = img_out(lur:rdr+pw, luc-pw:luc-1);
img_out_padding = [img_out_padding_a(:);img_out_padding_b(:);img_out_padding_c(:);img_out_padding_d(:)];

img_in_target = double( img_in_target(:) );
img_out_target = double( img_out_target(:) );
img_in_padding = double( img_in_padding(:) );
img_out_padding = double( img_out_padding(:) );

scr_in = (mean(img_in_target(:))-mean(img_in_padding(:)))/std(img_in_padding(:));
scr_out = (mean(img_out_target(:))-mean(img_out_padding(:)))/std(img_out_padding(:));

sg = scr_out/scr_in;
s = scr_out;