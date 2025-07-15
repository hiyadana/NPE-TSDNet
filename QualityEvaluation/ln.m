function Ln=ln(img)
h = size(img, 1);
w = size(img, 2);
img = double(img);

Lc = img(round(h/2),round(w/2));
L1 = img(8,8);
L2 = img(8,w-8);
L3 = img(h-8,w-8);
L4 = img(h-8,8);

Ln = (abs((Lc-L1)/Lc)+abs((Lc-L2)/Lc)+abs((Lc-L3)/Lc)+abs((Lc-L4)/Lc))*1/4;