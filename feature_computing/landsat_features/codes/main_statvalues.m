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
if ~exist('./STATS_GPU.ptx')
    cudabin='/usr/local/cuda-9.1/bin';    %add CUDA nvcc directory to system PATH (could be omitted if already in the PATH)
    path=getenv('PATH');
    path=[path,':',cudabin];
    setenv('PATH',path);
    system('nvcc -ptx STATS_GPU.cu'); 
end    

%Set workspace and directories
root='XXX';                               %please change to your local computer workspace (the parent directory of 'codes' folder)
folderdirfromname='data';
resultfoldername='result';
productname='DzB_Landsat';
filename_prefix='DzB_sample_band';
postfix='.tif';

%Set input parameters
default=0;                                %NoData value in GeoTiff file
WindowDim1=7;                             %the WindowDim for STD, Skewness & Kurtosis computing
WindowDim2=51;                            %the WindowDim for Median & Mean filters 

SplitDim1=1024;                           %Split the image to SplitDim*SplitDim blocks for GPU, using in statistical variables computing
SplitDim2=512;                            %Split the image to SplitDim*SplitDim blocks for GPU, using in filtering
BlockDim=16;                              %GPU threadBlock dimension
eps=1e-8;

%Median Filter Diameter, Gaussian Filter Sigma & Mean Filter Parameter
MFD1=11;                                  %median filter #1 diameter
MFD2=5;                                   %median filter #2 diameter
sigma1=4;                                 %gaussian filter sigma
MKR=5;                                    %mean filter kernel radius

%Down-sampling parameters
ori_resolution=30;                        %original resolution of input landsat images
mid_resolution=120;                       %mid-step downsampling resolution
tar_resolution=300;                       %target (final) resolution

%Spectral Variables
Bands={'1','2','3','4','5','6','7','dvi','ndvi','rvi','evi','vi01','vi02','vi03','savi','ndsi','si03'};

%Statistical Variables
Methods={'STD','CV','Skewness','Kurtosis'};

%Suppose Mapping Toolbox unavailable, using 'geotiffwrite.m' file instead
%images downloaded from Google Earth Engine may be splitted into several blocks
xlabels={'0000000000'};                             %postfixes of downloaded images from Google Earth Engine
ylabels={'0000000000'};                             %postfixes of downloaded images from Google Earth Engine
nrow=2085;                                          %nrow & ncol of the whole image
ncol=2176;
longitude_left=87.320467034;                        %left edge boundary of the whole image
longitude_right=87.9068872515;                      %right edge boundary of the whole image
latitude_top=47.1355572959;                         %top edge boundary of the whole image
latitude_bottom=46.5736610856;                      %bottom edge boundary of the whole image
bbox=[  longitude_left,  latitude_bottom;
 	    longitude_right,  latitude_top  ];          %longitude & latitude input of geotiffwrite.m file


for bi=1:length(Bands)
    Band=Bands{bi};
    disp('----------------------------------------------------------------------------------------');
    disp(' ');
    disp(['Calculating Band: ',Band,'...']);    
    disp(' ');
    t1=clock();
    %Calculate Vegetation Index
    Image=calcSpecVars(Band,root,folderdirfromname,productname,filename_prefix,postfix,xlabels,ylabels,nrow,ncol,default); 
    
    %Prepare Image Blocks
    disp('Prepare Image blocks...');
    WindowDim=WindowDim1;
    SplitDim=SplitDim1;
    SplitNumY=ceil(nrow/SplitDim);
    SplitNumX=ceil(ncol/SplitDim);
    SIDim=int32(SplitDim+WindowDim-1);
    SplitImageStore={};
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
            SIDim=int32(SplitDim+WindowDim-1);
            SplitImage=zeros([SIDim,SIDim],'single')+default;
            SplitImage(SI_ystart:SI_ystop,SI_xstart:SI_xstop)=Image(I_ystart:I_ystop,I_xstart:I_xstop);
            SplitImageStore{split_y,split_x}=SplitImage;
        end
    end
    clear Image;
    
    for mi=1:length(Methods)
    %Get Statistics Method
    method=Methods{mi};    
    disp(['Begin Part #1: Calculating Statistics: ',method]);
    WindowDim=WindowDim1;
    SplitDim=SplitDim1;
    SplitNumY=ceil(nrow/SplitDim);
    SplitNumX=ceil(ncol/SplitDim);
    SIDim=int32(SplitDim+WindowDim-1);
    ResultStore={};
    t1=clock();
    for split_y=1:SplitNumY
        for split_x=1:SplitNumX
            disp(['Calculating ',method,' in progress.... Texture Block (',num2str(split_y),',',num2str(split_x),')']);
            t5=clock();
            SplitImage=SplitImageStore{split_y,split_x};
            SplitImage=gpuArray(SplitImage);  
            MFBlock=zeros([SplitDim,SplitDim],'single','gpuArray')+default;
            if sum(sum(abs(SplitImage)))/single(SIDim*SIDim)==abs(default)
                ResultStore{split_y,split_x}=gather(MFBlock);
                SplitImageStore{split_y,split_x}=0;
                continue;
            end
            %Execute GPU Texture Program
            k=parallel.gpu.CUDAKernel('STATS_GPU.ptx','STATS_GPU.cu');
            blockSize=[BlockDim,BlockDim,1];
            bx=int32((SIDim+BlockDim-1)/BlockDim);
            by=int32((SIDim+BlockDim-1)/BlockDim);
            gridSize=[bx,by,1];
            k.ThreadBlockSize=blockSize;
            k.GridSize=gridSize;
            MFBlock=feval(k,MFBlock,SplitDim,SplitImage,SIDim,mi); 
            ResultStore{split_y,split_x}=gather(MFBlock);
            t6=clock();
            blockCalTime=etime(t6,t5);
            disp([method,' Block (',num2str(split_y),',',num2str(split_x),') Calculating Done.  Time Consuming = ',num2str(blockCalTime),'s']);
        end
    end
    t2=clock();
    calTime=etime(t2,t1);
    disp(['Calculating Done. Total Time Consuming = ',num2str(calTime),'s']);
    StatsImage=zeros([SplitNumY*SplitDim,SplitNumX*SplitDim],'single');
    for split_y=1:SplitNumY
        for split_x=1:SplitNumX  
            T_xstart=(split_x-1)*SplitDim+1;
            T_xstop=split_x*SplitDim;
            T_ystart=(split_y-1)*SplitDim+1;
            T_ystop=split_y*SplitDim;
            StatsImage(T_ystart:T_ystop,T_xstart:T_xstop)=ResultStore{split_y,split_x};
            ResultStore{split_y,split_x}={};
        end
    end
    StatsImage=StatsImage(1:nrow,1:ncol);
    clear ResultStore;
    
    disp('Begin Part #2: Mid-step Down-sampling...');
    scale=mid_resolution/ori_resolution;
    mid_nrow=round(nrow/scale);
    mid_ncol=round(ncol/scale);
    StatsImage=imresize(StatsImage,[mid_nrow,mid_ncol],'nearest'); 
    
    %Filtering The Result
    WindowDim=WindowDim2;  
    SplitDim=SplitDim2;
    SplitNumY=ceil(mid_nrow/SplitDim);
    SplitNumX=ceil(mid_ncol/SplitDim);
    SIDim=int32(SplitDim+WindowDim-1);
    SplitStatsStore={};
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
            SplitImage(SI_ystart:SI_ystop,SI_xstart:SI_xstop)=StatsImage(I_ystart:I_ystop,I_xstart:I_xstop);
            SplitStatsStore{split_y,split_x}=SplitImage;
        end
    end
    clear StatsImage;
    
    ResultStore={};
    t3=clock();
    disp('Begin Part #3: Filtering...');
    for split_y=1:SplitNumY
        for split_x=1:SplitNumX
            disp(['Executing Filters...    Block (',num2str(split_y),',',num2str(split_x),')']);
            t5=clock();
            SplitImage=SplitStatsStore{split_y,split_x};
            SplitImage=gpuArray(SplitImage);
            if sum(sum(abs(SplitImage)))/single(SIDim*SIDim)==abs(default)
                ResultStore{split_y,split_x}=gather(SplitImage(1+(WindowDim-1)/2:SplitDim+(WindowDim-1)/2,1+(WindowDim-1)/2:SplitDim+(WindowDim-1)/2));
                SplitStatsStore{split_y,split_x}={};
                continue;
            end
            MFBlock=medfilt2(SplitImage,[MFD1,MFD1]);
            GK=fspecial('gaussian',[(WindowDim-1)/2,(WindowDim-1)/2],sigma1);
            MFBlock=imfilter(MFBlock,GK,'conv','circular');
            MFBlock=medfilt2(MFBlock,[MFD2,MFD2]);
            MK=fspecial('disk',MKR);
            ResultBlock=imfilter(MFBlock,MK,'conv','circular');
            ResultBlock=ResultBlock(1+(WindowDim-1)/2:SplitDim+(WindowDim-1)/2,1+(WindowDim-1)/2:SplitDim+(WindowDim-1)/2);
            ResultStore{split_y,split_x}=gather(ResultBlock);
            SplitStatsStore{split_y,split_x}={};
            t6=clock();
            blockCalTime=etime(t6,t5);
            disp(['Filtering Block (',num2str(split_y),',',num2str(split_x),') Calculating Done.  Time Consuming = ',num2str(blockCalTime),'s']);
        end
    end
    t4=clock();
    calTime=etime(t4,t3);
    clear SplitStatsStore;
    disp(['Filters Calculating Done. Total Time Consuming = ',num2str(calTime),'s']);
    MeanFilter=zeros([SplitNumY*SplitDim,SplitNumX*SplitDim],'single');
    for split_y=1:SplitNumY
        for split_x=1:SplitNumX  
            T_xstart=(split_x-1)*SplitDim+1;
            T_xstop=split_x*SplitDim;
            T_ystart=(split_y-1)*SplitDim+1;
            T_ystop=split_y*SplitDim;
            MeanFilter(T_ystart:T_ystop,T_xstart:T_xstop)=ResultStore{split_y,split_x};
            ResultStore{split_y,split_x}={};
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
    folderdirto=[root,'/',resultfoldername,'/','Landsat_Statistical_Variables'];
    if ~exist(folderdirto,'dir')
        mkdir(folderdirto);
    end
    postfix='.tif';
    filedirto=[folderdirto,'/',filename_prefix,Band,'_',method,postfix];
    %[option, bbox] = make_option([bbox]);
    geotiffwrite(filedirto, bbox, TargetResult, 32);
    
    clear TargetResult;
    
    t2=clock();
    TotalTime=etime(t2,t1);
    disp(['Stats Method: ',method,' Finished!     Total Time Consuming = ',num2str(TotalTime),'s']);
    disp(' ');
    disp('****************************************************************************************');
    disp(' ');
    end
end
% delete(MP);
disp('----------------------------------------------------------------------------------------')
clear SplitImageStore;
disp('ALL DONE.');
