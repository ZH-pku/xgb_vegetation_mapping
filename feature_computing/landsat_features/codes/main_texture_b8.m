%@author: Heng Zhang
%last update: 03-31-2019

%Please CITE the following article when using the codes

%H. Zhang, A. Eziz, J. Xiao, S. Tao, S. Wang, Z. Tang, J. Zhu and J. Fang, 2019. High-resolution Vegetation Mapping Using eXtreme Gradient Boosting Based on Extensive Features. Remote Sensing.(submitted)
%emails: heng.zhang@pku.edu.cn, hengzhang.zhh@gmail.com; anwareziz@pku.edu.cn

%Computing Landsat Moving Window Statistical Variables
%Note: (1) NVIDIA GPU device is required in this program. 
%      (2) CUDA should be installed correctly, with 'nvcc -ptx XXX.cu' command available.
%      (3) If using Window system, please check whether graphic device TDR protection is
%      launched. Please modify the registry first. (search Google for detailed help)

clc;
clear;

% Prepare GPU environment
% if gpuDeviceCount==1
%     gpuDevice    
% else
%     gpuDevice(2)        %Select which GPU to use
% end
if gpuDeviceCount<1
    disp(['NO AVAILABLE NVIDIA GPU DEVICE FOUND!']);
    return
else
    MP=parpool(gpuDeviceCount)
end
spmd
    gpuDevice
end

%Complile NVCC ptx files
if ~exist('./CalculateTexture_GPU.ptx')
    cudabin='/usr/local/cuda-9.1/bin';
    path=getenv('PATH');
    path=[path,':',cudabin];
    setenv('PATH',path);
    system('nvcc -ptx CalculateTexture_GPU.cu'); 
end

%Set workspace and directories
root='XXX';                               %please change to your local computer workspace (the parent directory of 'codes' folder)
folderdirfromname='data';
resultfoldername='result';
productname='DzB_Landsat';
filename_prefix='DzB_sample_band';
postfix='.tif';
filename=[filename_prefix,'8'];

%Set input parameters
d=1;                                      %shift distance to center (Haralick, R.M. et al., 1973)
default=0;                                %NoData value in GeoTiff file
WindowDim1=13;                            %the moving window dimension for texture computing, an odd recommended
WindowDim2=91;                            %the moving window dimension for median and mean filters
bool_mean=1;                              %0 represents for the range of texture values in four directions; 
                                          %1 represents for the average of texture value in four directions. 
                                          %(See Haralick, R.M. et al., 1973 Page 615)
Ng=32;                                    %number of grayscale level (Haralick, R.M. et al., 1973)
       
SplitDim1=1024;                           %Split the image to SplitDim*SplitDim blocks for GPU, using in statistical variables computing
SplitDim2=512;                            %Split the image to SplitDim*SplitDim blocks for GPU, using in filtering
BlockDim=16;                              %GPU threadBlock dimension

%NOTICE: Before executing, please MAKE SURE that d, Ng, default,
%WindowDim1, BlockDim are the same as D, NG, DEFAULT, WINDOWDIM, BLOCKDIM,
%respectively. If not equal, to avoid mistakes, after value revision please
%DELETE 'CalculateTexture_GPU.ptx' file so that nvcc will compile a new
%one instead.

%Median Filter Diameter, Gaussian Filter Sigma & Mean Filter Parameter
MFD1=15;                                  %median filter #1 diameter
MFD2=9;                                   %median filter #2 diameter
sigma1=3;                                 %gaussian filter sigma
MKR=10;                                   %mean filter kernel radius

%Down-sampling Parameters
ori_resolution=15;                        %original resolution of input landsat images (band 8 is 15 m resolution)
mid_resolution=60;                        %mid-step downsampling resolution
tar_resolution=300;                       %target (final) resolution

%Suppose Mapping Toolbox unavailable, using 'geotiffwrite.m' file instead
%images downloaded from Google Earth Engine may be splitted into several blocks
xlabels={'0000000000'};                             %postfixes of downloaded images from Google Earth Engine
ylabels={'0000000000'};                             %postfixes of downloaded images from Google Earth Engine
nrow=4170;                                          %nrow & ncol of the whole image
ncol=4352;
longitude_left=87.320467034;                        %left edge boundary of the whole image
longitude_right=87.9068872531;                      %right edge boundary of the whole image
latitude_top=47.135557297;                         %top edge boundary of the whole image
latitude_bottom=46.5736610851;                      %bottom edge boundary of the whole image
bbox=[  longitude_left,  latitude_bottom;
 	    longitude_right,  latitude_top  ];          %longitude & latitude input of geotiffwrite.m file

%Calculate quantiles
disp('Calculate Quantiles...');
%the quantile reference file could be appointed alternatively or use the whole image instead
% refQFile=[root,'/',folderdirfromname,'/',productname,'/',filename,postfix];
% QImage=single(imread(refQFile));
QImage=readBlockImage(root,folderdirfromname,productname,filename,postfix,xlabels,ylabels,nrow,ncol,default);
QImage(isnan(QImage))=default;
p=single(linspace(0,1,Ng+1));
p=p(2:Ng);
Q=quantile(QImage(QImage~=default),p);
clear QImage;

for method=1:13
    t1=clock();
    disp(' ');
    disp(['Processing Method = ',num2str(method)]);
    disp(' ');
    disp('Reading Source File...')
    Image=readBlockImage(root,folderdirfromname,productname,filename,postfix,xlabels,ylabels,nrow,ncol,default);
    WindowDim=WindowDim1;          %Recommand an odd
    SplitDim=SplitDim1;
    SplitNumY=ceil(nrow/SplitDim);
    SplitNumX=ceil(ncol/SplitDim);
    disp('Prepare Image blocks...');
    SplitImageStore={};
    SIDim=int32(SplitDim+WindowDim-1);
    for split_y=1:SplitNumY
        for split_x=1:SplitNumX
            disp(['Copying....   Block (',num2str(split_y),',',num2str(split_x),')']);
            SI_ystart=1;SI_xstart=1;
            I_ystart=max((split_y-1)*SplitDim+1-(WindowDim-1)/2,1);
            if I_ystart==1
                SI_ystart=(WindowDim-1)/2+1;
            end
            I_ystop=min(split_y*SplitDim+(WindowDim-1)/2,nrow);
            SI_ystop=SI_ystart+(I_ystop-I_ystart);
            I_xstart=max((split_x-1)*SplitDim+1-(WindowDim-1)/2,1);
            if I_xstart==1
                SI_xstart=(WindowDim-1)/2+1;
            end
            I_xstop=min(split_x*SplitDim+(WindowDim-1)/2,ncol);
            SI_xstop=SI_xstart+(I_xstop-I_xstart);
            SplitImage=zeros([SIDim,SIDim],'single')+default;
            SplitImage(SI_ystart:SI_ystop,SI_xstart:SI_xstop)=Image(I_ystart:I_ystop,I_xstart:I_xstop);
            SplitImageStore{split_y,split_x}=SplitImage;
        end
    end
    clear Image;

    ResultStore={};
    t3=clock();
    disp('Begin Part #1: Texture Calculating...');
    for split_y=1:SplitNumY
        for split_x=1:SplitNumX
            disp(['Calculating Texture in progress.... Texture Block (',num2str(split_y),',',num2str(split_x),')']);
            t5=clock();
            SplitImage=SplitImageStore{split_y,split_x};
            SplitImage=gpuArray(SplitImage);  
            TextureBlock=zeros([SplitDim,SplitDim],'single','gpuArray')+default;
            if sum(sum(abs(SplitImage)))/single(SIDim*SIDim)==abs(default)
                ResultStore{split_y,split_x}=gather(TextureBlock);
                SplitImageStore{split_y,split_x}={};
                continue;
            end
            %Execute GPU Texture Program
            k=parallel.gpu.CUDAKernel('CalculateTexture_GPU.ptx','CalculateTexture_GPU.cu');
            blockSize=[BlockDim,BlockDim,1];
            bx=int32((SIDim+BlockDim-1)/BlockDim);
            by=int32((SIDim+BlockDim-1)/BlockDim);
            gridSize=[bx,by,1];
            k.ThreadBlockSize=blockSize;
            k.GridSize=gridSize;
            TextureBlock=feval(k,TextureBlock,SplitDim,SplitImage,SIDim,Q,method,bool_mean); 
            ResultStore{split_y,split_x}=gather(TextureBlock);
            SplitImageStore{split_y,split_x}={};
            t6=clock();
            blockCalTime=etime(t6,t5);
            disp(['Texture Block (',num2str(split_y),',',num2str(split_x),') Calculating Done.  Time Consuming = ',num2str(blockCalTime),'s']);
        end
    end
    t4=clock();
    calTime=etime(t4,t3);

    disp(['Texture Calculating Done. Total Time Consuming = ',num2str(calTime),'s']);
    Texture=zeros([SplitNumY*SplitDim,SplitNumX*SplitDim],'single');
    for split_y=1:SplitNumY
        for split_x=1:SplitNumX  
            T_xstart=(split_x-1)*SplitDim+1;
            T_xstop=split_x*SplitDim;
            T_ystart=(split_y-1)*SplitDim+1;
            T_ystop=split_y*SplitDim;
            Texture(T_ystart:T_ystop,T_xstart:T_xstop)=ResultStore{split_y,split_x};
            ResultStore{split_y,split_x}={};
        end
    end
    Texture=Texture(1:nrow,1:ncol);
    clear ResultStore;
    
    disp('Begin Part #2: Mid-step Down-sampling...');
    scale=mid_resolution/ori_resolution;
    mid_nrow=round(nrow/scale);
    mid_ncol=round(ncol/scale);
    Texture=imresize(Texture,[mid_nrow,mid_ncol],'nearest');  
    
    WindowDim=WindowDim2;
    SplitDim=SplitDim2;
    SplitNumY=ceil(mid_nrow/SplitDim);
    SplitNumX=ceil(mid_ncol/SplitDim);    
    disp('Prepare Texture blocks...');
    SIDim=int32(SplitDim+WindowDim-1);
    SplitTextureStore={};
    for split_y=1:SplitNumY
        for split_x=1:SplitNumX
            disp(['Copying....   Block (',num2str(split_y),',',num2str(split_x),')']);
            SI_ystart=1;SI_xstart=1;
            I_ystart=max((split_y-1)*SplitDim+1-(WindowDim-1)/2,1);
            if I_ystart==1
                SI_ystart=(WindowDim-1)/2+1;
            end
            I_ystop=min(split_y*SplitDim+(WindowDim-1)/2,mid_nrow);
            SI_ystop=SI_ystart+(I_ystop-I_ystart);
            I_xstart=max((split_x-1)*SplitDim+1-(WindowDim-1)/2,1);
            if I_xstart==1
                SI_xstart=(WindowDim-1)/2+1;
            end
            I_xstop=min(split_x*SplitDim+(WindowDim-1)/2,mid_ncol);
            SI_xstop=SI_xstart+(I_xstop-I_xstart);
            SIDim=int32(SplitDim+WindowDim-1);
            SplitImage=zeros([SIDim,SIDim],'single')+default;
            SplitImage(SI_ystart:SI_ystop,SI_xstart:SI_xstop)=Texture(I_ystart:I_ystop,I_xstart:I_xstop);
            SplitTextureStore{split_y,split_x}=SplitImage;
        end
    end
    clear Texture;
    
    ResultStore={};
    t3=clock();
    disp('Begin Part #3: Filtering...');
    for split_y=1:SplitNumY
        for split_x=1:SplitNumX
            disp(['Executing Filters...    Block (',num2str(split_y),',',num2str(split_x),')']);
            t5=clock();
            SplitImage=SplitTextureStore{split_y,split_x};
            SplitImage=gpuArray(SplitImage);
            if sum(sum(abs(SplitImage)))/single(SIDim*SIDim)==abs(default)
                ResultStore{split_y,split_x}=gather(SplitImage(1+(WindowDim-1)/2:SplitDim+(WindowDim-1)/2,1+(WindowDim-1)/2:SplitDim+(WindowDim-1)/2));
                SplitTextureStore{split_y,split_x}=0;
                continue;
            end
            MFBlock=medfilt2(SplitImage,[MFD1,MFD1]);
%             GK=fspecial('gaussian',[(WindowDim-1)/2,(WindowDim-1)/2],sigma1);
%             MFBlock=imfilter(MFBlock,GK,'conv','circular');
%             MFBlock=medfilt2(MFBlock,[MFD2,MFD2]);
            MK=fspecial('disk',MKR);
            ResultBlock=imfilter(MFBlock,MK,'conv','circular');
            ResultBlock=ResultBlock(1+(WindowDim-1)/2:SplitDim+(WindowDim-1)/2,1+(WindowDim-1)/2:SplitDim+(WindowDim-1)/2);
            ResultStore{split_y,split_x}=gather(ResultBlock);
            SplitTextureStore{split_y,split_x}=0;
            t6=clock();
            blockCalTime=etime(t6,t5);
            disp(['Filtering Block (',num2str(split_y),',',num2str(split_x),') Calculating Done.  Time Consuming = ',num2str(blockCalTime),'s']);
        end
    end
    t4=clock();
    calTime=etime(t4,t3);
    clear SplitTextureStore;
    disp(['Filters Calculating Done. Total Time Consuming = ',num2str(calTime),'s']);
    MeanFilter=zeros([SplitNumY*SplitDim,SplitNumX*SplitDim],'single');
    for split_y=1:SplitNumY
        for split_x=1:SplitNumX  
            T_xstart=(split_x-1)*SplitDim+1;
            T_xstop=split_x*SplitDim;
            T_ystart=(split_y-1)*SplitDim+1;
            T_ystop=split_y*SplitDim;
            MeanFilter(T_ystart:T_ystop,T_xstart:T_xstop)=ResultStore{split_y,split_x};
            ResultStore{split_y,split_x}=0;
        end
    end
    clear ResultStore;
    MeanFilter=MeanFilter(1:mid_nrow,1:mid_ncol);

    disp('Begin Part #4: Down-sampling...');
    scale=tar_resolution/ori_resolution;
    tar_nrow=round(nrow/scale);
    tar_ncol=round(ncol/scale);
    TargetResult=imresize(MeanFilter,[tar_nrow,tar_ncol],'nearest');
    clear MeanFilter;

    disp('Begin Part #5: Writing GeoTIFF Files...');
    folderdirto=[root,'/',resultfoldername,'/','Landsat_Band8_Textures'];
    if ~exist(folderdirto,'dir')
        mkdir(folderdirto);
    end
    postfix='.tif';
    filedirto=[folderdirto,'/',filename,'_texture_M',num2str(method),postfix];
    %[option, bbox] = make_option([bbox]);
    geotiffwrite(filedirto, bbox, TargetResult, 32);    

    t2=clock();
    TotalTime=etime(t2,t1);
    disp(['Method = ',num2str(method),' All Finished!     Total Time Consuming = ',num2str(TotalTime),'s']);
    disp(' ');
    disp('****************************************************************************************');
    disp(' ');
end
%delete(MP);
clear SplitImageStore;
disp('ALL DONE.');