 I=rgb2gray(imread('cell115.jpg'));
 X = fft2(I);
 f_shift=fftshift(log(abs(X) + 1)); imshow(f_shift,[]);
 f=edge(f_shift,"Sobel");
%  imshow(f, []); hold on;

 [H,theta,rho] = hough(f);
 P = houghpeaks(H,5,'threshold',ceil(0.3*max(H(:))));
 lines = houghlines(f,theta,rho,P,'FillGap',5,'MinLength',15);
 
 for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   %plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
 end

 [a,b]=size(f);
 mask=ones(a,b);
 for i=1:length(lines)
     xy = [lines(i).point1; lines(i).point2];
     for y=xy(1,1):xy(2,1)
         for x=xy(1,2):xy(2,2)
            mask(x,y)=0;
         end
     end
 end

 f2=f_shift.*mask;
%  figure;imshow(f2,[]);
 image=ifft2(ifftshift(f2));
 figure;imshow(image);
