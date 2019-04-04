function colormap=cvtColorSpace(RGBImage,ColorSpace,default)
    if ~exist('./colorspace.m')
        disp('No available converting functions found!');
        disp('Only RGB -> HSV, YCbCr can be used.');
    end
    rgbsize=size(RGBImage);
    RGB=zeros(rgbsize,'double');
    temp=RGBImage(:,:,1);
    bool_default=temp;
    bool_default(temp==default)=1;
    bool_default(temp~=default)=0; 
    for i=1:rgbsize(3)
        temp=double(RGBImage(:,:,i));
        tempmin=min(min(temp(bool_default==0)));
        tempmax=max(max(temp(bool_default==0)));
        temp=255*(temp-tempmin)./(tempmax-tempmin);
        temp(bool_default==1)=default;
        RGB(:,:,i)=temp;
    end
    switch ColorSpace
        case 'hsv'
            colormap=rgb2hsv(RGB);
        case 'ycbcr'
            colormap=rgb2ycbcr(RGB);
        case 'lab'
            csstr='Lab<-RGB';
            colormap=colorspace(csstr,RGB);
        case 'xyz'
            csstr='XYZ<-RGB';
            colormap=colorspace(csstr,RGB); 
        case 'yiq'
            csstr='YIQ<-RGB';
            colormap=colorspace(csstr,RGB); 
        case 'yuv'
            csstr='YUV<-RGB';
            colormap=colorspace(csstr,RGB);
        case 'lch'
            csstr='LCH<-RGB';
            colormap=colorspace(csstr,RGB);
        otherwise
            disp('Wrong Input Color Space!');
    end
    colormap=single(colormap);
    cmsize=size(colormap);
    for i=1:length(cmsize(3))
        temp=colormap(:,:,i);
        temp(bool_default==1)=default;
        colormap(:,:,i)=temp;
    end
end