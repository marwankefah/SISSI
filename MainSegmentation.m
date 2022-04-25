imagefiles = dir('*.jpg');
nfiles = length(imagefiles); 


for i=1:nfiles
    figure(i); subplot(3,3,1);
    I=rgb2gray(imread(imagefiles(i).name));
    imshow(I);
    
    I_ilumination=ilumination_correction(I);
    subplot(3,3,2);imshow(I_ilumination);
    I=adapthisteq(I_ilumination,'NumTiles',[16 16]);
    subplot(3,3,3);imshow(I);
    
    S=EGT_Segmentation(imagefiles(i).name, 500, 30, 1);
    SE=strel('disk',7);S2=imopen(S,SE);

    foreground=int16(I).*int16(S2);
    subplot(3,3,4);imshow(foreground,[0 255]);

    background=int16(I).*int16(imcomplement(S2));
    subplot(3,3,5);imshow(background,[0 255]);

    res_glog_kong=glog_kong(I,S2,10,0.03);
    sum(sum(res_glog_kong))
    SE=strel('disk',3);res_glog_kong=imdilate(res_glog_kong,SE);
    
    [a,b]=size(res_glog_kong); A=zeros(a,b,3);
    A(:,:,1)=int16(res_glog_kong)*255;
    
    subplot(3,3,6);hold on; imshow(A);hb=imshow(I,[]); hb.AlphaData = 0.4; 
end