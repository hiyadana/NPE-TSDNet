function [RMSE,p]=psnr(img1,img2,data_max)
img1 = double(img1);
img2 = double(img2);

MSE=mse(img1-img2);
RMSE = sqrt(MSE);
p=20*log10(data_max/RMSE);