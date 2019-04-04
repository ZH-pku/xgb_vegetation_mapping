
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>

#define DEFAULT 0
#define WINDOWDIM 7
#define BLOCKDIM 16
#define EPS 1e-4
#define WRONGNUM -9999

__device__ int calInd(int Dim, int y, int x)
{
	return y*Dim + x;
}

__device__ float2 cal_mu_std(float *SubSplitImage, int SSIDim, int ix, int iy, int flag)            //flag = 0 only calculate mean, flag = 1 calculate mean and std
{
	float2 ans = { 0.0 };                           //ans.x is mean, ans.y is standard deviation
	int count = 0;
	float value;
	for (int j = iy; j < WINDOWDIM + iy; j++)
	{
		for (int i = ix; i < WINDOWDIM + ix; i++)
		{
			value = SubSplitImage[calInd(SSIDim, j, i)];
			if (value != DEFAULT)
			{
				ans.x += value;
				count++;
			}				
		}
	}
	ans.x = ans.x / count;
	if (flag == 1)
	{
		count = 0;
		for (int j = iy; j < WINDOWDIM + iy; j++)
		{
			for (int i = ix; i < WINDOWDIM + ix; i++)
			{
				value = SubSplitImage[calInd(SSIDim, j, i)];
				if (value != DEFAULT)
				{
					ans.y += (value - ans.x)*(value - ans.x);
					count++;
				}
			}
		}
		ans.y = sqrt(ans.y / count + EPS);
	}
	return ans;
}

__device__ float calMoment3(float *SubSplitImage, int SSIDim, int ix, int iy)
{
	float moment = 0.0;
	float value, temp;
	int count = 0;
	float2 mu_std = cal_mu_std(SubSplitImage, SSIDim, ix, iy, 1);
	for (int j = iy; j < WINDOWDIM + iy; j++)
	{
		for (int i = ix; i < WINDOWDIM + ix; i++)
		{
			value = SubSplitImage[calInd(SSIDim, j, i)];
			if (value != DEFAULT)
			{
				temp = (value - mu_std.x) / mu_std.y;
				moment += temp*temp*temp;
				count++;
			}
		}
	}
	moment = moment / count;
	return moment;
}

__device__ float calMoment4(float *SubSplitImage, int SSIDim, int ix, int iy)
{
	float moment = 0.0;
	float value, temp;
	int count = 0;
	float2 mu_std = cal_mu_std(SubSplitImage, SSIDim, ix, iy, 1);
	for (int j = iy; j < WINDOWDIM + iy; j++)
	{
		for (int i = ix; i < WINDOWDIM + ix; i++)
		{
			value = SubSplitImage[calInd(SSIDim, j, i)];
			if (value != DEFAULT)
			{
				temp = (value - mu_std.x) / mu_std.y;
				moment += temp*temp*temp*temp;
				count++;
			}
		}
	}
	moment = moment / count;
	return moment;
}

__device__ void copyImage(float *SplitImage, int SIDim, float *SubSplitImage, int SSIDim, int x, int y, int ix, int iy)
{
	SubSplitImage[calInd(SSIDim, iy, ix)] = SplitImage[calInd(SIDim, x, y)];
	if (ix < WINDOWDIM - 1)
		SubSplitImage[calInd(SSIDim, iy, ix + BLOCKDIM)] = SplitImage[calInd(SIDim, x + BLOCKDIM, y)];
	if (iy < WINDOWDIM - 1)
		SubSplitImage[calInd(SSIDim, iy + BLOCKDIM, ix)] = SplitImage[calInd(SIDim, x, y + BLOCKDIM)];
	if (ix < WINDOWDIM - 1 && iy < WINDOWDIM - 1)
		SubSplitImage[calInd(SSIDim, iy + BLOCKDIM, ix + BLOCKDIM)]
		= SplitImage[calInd(SIDim, x + BLOCKDIM, y + BLOCKDIM)];
	__syncthreads();
}


__global__ void gpuMeanFilter(float *MFImage, int MFDim, float *SplitImage, int SIDim, int mi)               //mi is the same as stats_bands_main.m
{
	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	if (x >= MFDim || y >= MFDim)
		return;
	__shared__ float SubSplitImage[(BLOCKDIM + WINDOWDIM - 1)*(BLOCKDIM + WINDOWDIM - 1)];
	int SSIDim = BLOCKDIM + WINDOWDIM - 1;
	copyImage(SplitImage, SIDim, SubSplitImage, SSIDim, x, y, ix, iy);
	if (SplitImage[calInd(SIDim, x + (WINDOWDIM - 1) / 2, y + (WINDOWDIM - 1) / 2)] == DEFAULT)
		MFImage[calInd(MFDim, x, y)] = DEFAULT;
	else
	{
		if (mi == 1)
		{
			float2 mu_std = cal_mu_std(SubSplitImage, SSIDim, ix, iy, 1);
			MFImage[calInd(MFDim, x, y)] = mu_std.y;
		}
		else if (mi == 2)
		{
			float2 mu_std = cal_mu_std(SubSplitImage, SSIDim, ix, iy, 1);
			MFImage[calInd(MFDim, x, y)] = mu_std.y / (mu_std.x + EPS);
		}
		else if (mi == 3)
			MFImage[calInd(MFDim, x, y)] = calMoment3(SubSplitImage, SSIDim, ix, iy);
		else if (mi == 4)
			MFImage[calInd(MFDim, x, y)] = calMoment4(SubSplitImage, SSIDim, ix, iy);
		else
			MFImage[calInd(MFDim, x, y)] = WRONGNUM;
	}
}