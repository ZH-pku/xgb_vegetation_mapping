
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>

#define D 1                           //Please set it to d in texture_main.m manually
#define NG 32                         //Please set it to Ng in texture_main.m manually
#define DEFAULT 0                     //Please set it to default in texture_main.m manually
#define WINDOWDIM 9                   //Please set it to WindowDim in texture_main.m manually
#define BlOCKDIM 16                   //Please set it to BlockDim in texture_main.m manually
#define EPS 1e-8

__device__ int calInd(int Dim, int y, int x)
{
	return y*Dim + x;
}

__device__ float calAvgOrRange(float *X, int n, int bool_mean)
{
	float ans = 0.0;
	if (bool_mean == 1)
	{
		for (int i = 0; i < n; i++)
			ans += X[i];
	}
	else
	{
		float max = X[0];
		float min = max;
		for (int i = 0; i < n; i++)
		{
			if (X[i] > max)
				max = X[i];
			if (X[i] < min)
				min = X[i];
		}
		ans = max - min;
	}
	return ans;
}

__device__ float calPx(unsigned char *SDM, float sum, int i)
{
	float px = 0.0;
	for (int k = 0; k < NG; k++)
	{
		px += (float)SDM[calInd(NG, k, i)] / sum;
	}
	return px;
}

__device__ float calPy(unsigned char *SDM, float sum, int j)
{
	float py = 0.0;
	for (int k = 0; k < NG; k++)
	{
		py += (float)SDM[calInd(NG, j, k)] / sum;
	}
	return py;
}

__device__ float calPx_plus_y(unsigned char *SDM, float sum, int k)               //k is the same as the paper listed
{
	k = k - 2;                              //k = 2, 3, ..., 2NG
	float pxy = 0.0;
	int i, j;
	int lowerlimit, upperlimit;
	if (k < NG)
	{
		lowerlimit = 0;
		upperlimit = k + 1;
	}		
	else
	{
		lowerlimit = k - NG + 1;
		upperlimit = NG;
	}
	for (j = lowerlimit; j < upperlimit; j++)
	{
		i = k - j;
		pxy += (float)SDM[calInd(NG, j, i)] / sum;
	}
	return pxy;
}

__device__ float calPx_minus_y(unsigned char *SDM, float sum, int k)
{
	float pxy = 0.0;
	int i, j;
	int lowerlimit, upperlimit;
	lowerlimit = 0;
	upperlimit = NG - k;
	for (j = lowerlimit; j < upperlimit; j++)
	{
		i = j + k;
		pxy += (float)SDM[calInd(NG, j, i)] / sum;
	}
	lowerlimit = k;
	upperlimit = NG;
	for (j = lowerlimit; j < upperlimit; j++)
	{
		i = j - k;
		pxy += (float)SDM[calInd(NG, j, i)] / sum;
	}
	return pxy;
}

__device__ float2 cal_mu_std_x(unsigned char *SDM, float sum, int flag)            //flag = 0 only calculate mean, flag = 1 calculate mean and std
{
	float px[NG] = { 0.0 };
	float2 ans;                           //ans.x is mean, ans.y is standard deviation
	ans.x = 0; ans.y = 0;
	for (int i = 0; i < NG; i++)
	{
		px[i] = calPx(SDM, sum, i);
	}
	for (int i = 0; i < NG; i++)
		ans.x += px[i];
	ans.x = ans.x / NG;
	if (flag == 1)
	{
		for (int i = 0; i < NG; i++)
		{
			ans.y += (px[i] - ans.x)* (px[i] - ans.x);
		}
		ans.y = sqrt(ans.y / NG);
	}
	return ans;
}

__device__ float2 cal_mu_std_y(unsigned char *SDM, float sum, int flag)            //flag = 0 only calculate mean, flag = 1 calculate mean and std
{
	float py[NG] = { 0.0 };
	float2 ans;                           //ans.x is mean, ans.y is standard deviation
	ans.x = 0; ans.y = 0;
	for (int j = 0; j < NG; j++)
	{
		py[j] = calPy(SDM, sum, j);
	}
	for (int j = 0; j < NG; j++)
		ans.x += py[j];
	ans.x = ans.x / NG;
	if (flag == 1)
	{
		for (int j = 0; j < NG; j++)
		{
			ans.y += (py[j] - ans.x) * (py[j] - ans.x);
		}
		ans.y = sqrt(ans.y / NG);
	}
	return ans;
}

__device__ float calmu(unsigned char *SDM, float sum)
{
	float px;
	float mu = 0.0;
	for (int j = 0; j < NG; j++)
	{
		for (int i = 0; i < NG; i++)
		{
			px = calPx(SDM, sum, i);
			mu += px*i;
		}
	}
	return mu;
}

__device__ float calHXY(unsigned char *SDM, float sum)
{
	float HXY = 0.0;
	float p;
	for (int i = 0; i < NG*NG; i++)
	{
		p = (float)SDM[i] / sum;
		HXY -= p*log(p + EPS);
	}
	return HXY;
}

__device__ float calHXY1(unsigned char *SDM, float sum)
{
	float HXY1 = 0.0;
	float p;
	float px[NG];
	float py[NG];
	for (int i = 0; i < NG; i++)
		px[i] = calPx(SDM, sum, i);
	for (int j = 0; j < NG; j++)
		py[j] = calPy(SDM, sum, j);
	for (int j = 0; j < NG; j++)
	{
		for (int i = 0; i < NG; i++)
		{
			p = (float)SDM[calInd(NG, j, i)] / sum;
			HXY1 -= p*log(px[i] * py[j] + EPS);
		}
	}
	return HXY1;
}

__device__ float calHXY2(unsigned char *SDM, float sum)
{
	float HXY2 = 0.0;
	float p;
	float px[NG];
	float py[NG];
	for (int i = 0; i < NG; i++)
		px[i] = calPx(SDM, sum, i);
	for (int j = 0; j < NG; j++)
		py[j] = calPy(SDM, sum, j);
	for (int j = 0; j < NG; j++)
	{
		for (int i = 0; i < NG; i++)
		{
			p = px[i] * py[j];
			HXY2 -= p*log(p + EPS);
		}
	}
	return HXY2;
}

__device__ float calHX(unsigned char *SDM, float sum)
{
	float HX = 0.0;
	float p;
	for (int i = 0; i < NG; i++)
	{
		p = calPx(SDM, sum, i);
		HX -= p*log(p + EPS);
	}
	return HX;
}

__device__ float calHY(unsigned char *SDM, float sum)
{
	float HY = 0.0;
	float p;
	for (int j = 0; j < NG; j++)
	{
		p = calPy(SDM, sum, j);
		HY -= p*log(p + EPS);
	}
	return HY;
}

__device__ float calTexture(unsigned char *SDM, int method)
{
	float texture = 0.0;
	float sum = 0.0;
	float p = 0.0;
	for (int i = 0; i < NG*NG; i++)
		sum += (float)SDM[i];
	if (method == 0)
	{
		texture = sum;
	}
	else if (method == 1)                                       //Angular Second Moment
	{
		if (sum == 0)
			texture = 0;
		else
		{
			for (int i = 0; i < NG*NG; i++)
			{
				p = (float)SDM[i] / sum;
				texture += p*p;
			}
		}		
	}
	else if (method == 2)                                       //Contrast
	{
		if (sum == 0)
			texture = 0;
		else
		{
			for (int n = 0; n < NG; n++)
			{				
				texture += n*n*calPx_minus_y(SDM, sum, n);
			}
		}
	}
	else if (method == 3)                                       //Correlation
	{
		if (sum == 0)
			texture = 0;
		else
		{
			float2 mustd_x = cal_mu_std_x(SDM, sum, 1);
			float2 mustd_y = cal_mu_std_y(SDM, sum, 1);
			for (int j = 0; j < NG; j++)
			{
				for (int i = 0; i < NG; i++)
				{
					p += i*j*(float)SDM[calInd(NG, j, i)] / sum;
				}
			}
			texture = (p - mustd_x.x * mustd_y.x) / (mustd_x.y * mustd_y.y);
		}
	}
	else if (method == 4)                                       //Sum of Squares: Variance
	{
		if (sum == 0)
			texture = 0;
		else
		{
			float mu = calmu(SDM, sum);
			for (int j = 0; j < NG; j++)
			{
				for (int i = 0; i < NG; i++)
				{
					p = (float)SDM[calInd(NG, j, i)] / sum;
					texture += (i - mu)*(i - mu)*p;
				}
			}
		}
	}
	else if (method == 5)                                       //Inverse Difference Moment
	{
		if (sum == 0)
			texture = 0;
		else
		{
			for (int j = 0; j < NG; j++)
			{
				for (int i = 0; i < NG; i++)
				{
					p = (float)SDM[calInd(NG, j, i)] / sum;
					texture += p / (1 + (i - j)*(i - j));
				}
			}
		}
	}
	else if (method == 6)                                       //Sum Average
	{
		if (sum == 0)
			texture = 0;
		else 
		{
			for (int k = 2; k <= 2 * NG; k++)
			{
				texture += k*calPx_plus_y(SDM, sum, k);
			}			
		}
	}
	else if (method == 7)                                       //Sum Variance
	{
		if (sum == 0)
			texture = 0;
		else
		{
			float pxy[2 * NG - 1];
			float f8 = 0.0;
			for (int k = 2; k <= 2 * NG; k++)
			{
				p = calPx_plus_y(SDM, sum, k);
				pxy[k - 2] = p;
				f8 -= p*log(p + EPS);
			}
			for (int k = 2; k <= 2 * NG; k++)
			{
				texture += (k - f8)*(k - f8)*pxy[k - 2];
			}
		}
	}
	else if (method == 8)                                       //Sum Entropy
	{
		if (sum == 0)
			texture = 0;
		else
		{
			for (int k = 2; k <= 2 * NG; k++)
			{
				p = calPx_plus_y(SDM, sum, k);
				texture -= p*log(p + EPS);
			}
		}
	}
	else if (method == 9)                                       //Entropy
	{
		if (sum == 0)
			texture = 0;
		else
		{
			texture = calHXY(SDM, sum);
		}
	}
	else if (method == 10)                                      //Difference Variance
	{
		if (sum == 0)
			texture = 0;
		else
		{
			float pxy[NG];
			float mean = 0.0;
			for (int k = 0; k < NG; k++)
			{
				pxy[k] = calPx_minus_y(SDM, sum, k);
				mean += pxy[k];
			}
			mean = mean / NG;
			for (int k = 0; k < NG; k++)
			{
				texture += (pxy[k] - mean)*(pxy[k] - mean);
			}
			texture = texture / NG;
		}
	}
	else if (method == 11)                                      //Difference Entropy
	{
		if (sum == 0)
			texture = 0;
		else
		{
			for (int k = 0; k < NG; k++)
			{
				p = calPx_minus_y(SDM, sum, k);
				texture -= p*log(p + EPS);
			}			
		}
	}
	else if (method == 12)                                      //Information Mesures of Correlation
	{
		if (sum == 0)
			texture = 0;
		else
		{
			float HX = calHX(SDM, sum);
			float HY = calHY(SDM, sum);
			float H;
			if (HX >= HY)
				H = HX;
			else
				H = HY;
			texture = (calHXY(SDM, sum) - calHXY1(SDM, sum)) / H;
		}
	}
	else if (method == 13)                                      //Information Mesures of Correlation
	{
		if (sum == 0)
			texture = 0;
		else
		{
			texture = 1 - exp(-2.0*(calHXY2(SDM, sum) - calHXY(SDM, sum)));
			if (texture < 0)
				texture = 0;
			else
				texture = sqrt(texture);
		}
	}
	else if (method == 14)                                      //Maximal Correlation Coefficient
	{
		/*
		if (sum == 0)
			texture = 0;
		else
		{
			float Q[NG*NG];
			float q, pik, pjk, pxi, pyk;
			for (int j = 0; j < NG; j++)
			{
				for (int i = 0; i < NG; i++)
				{
					pxi = calPx(SDM, sum, i);
					for (int k = 0; k < NG; k++)
					{
						pik= (float)SDM[calInd(NG, k, i)] / sum;
						pjk = (float)SDM[calInd(NG, k, j)] / sum;
						pyk = calPy(SDM, sum, k);
						q += (pik*pjk) / (pxi*pyk + EPS);
					}
					Q[calInd(NG, j, i)] = q;
				}
			}
			//Next are solving Q.TQ second largest eigenvalue, supposed to use QR decomposing. 
			//It hardly available for a single thread in GPU.
		}
		*/

	}
	return texture;
}

__device__ void updateSDM(unsigned char *SDM, int value1, int value2)
{
	SDM[calInd(NG, value1, value2)] += 1;
}

__device__ int convert2scale(float value, float *Q)
{
	int rank = 0;
	while (Q[rank] < value && rank < NG - 1)
		rank += 1;
	return rank;
}

__device__ void copyAndConvertImage(float *SplitImage, int SIDim, int *SubSplitImage, int SSIDim, float *Q,
																int x, int y, int ix, int iy)
{
	SubSplitImage[calInd(SSIDim, iy, ix)] = convert2scale(SplitImage[calInd(SIDim, x, y)], Q);
	if (ix < WINDOWDIM - 1)
		SubSplitImage[calInd(SSIDim, iy, ix + BlOCKDIM)] = convert2scale(SplitImage[calInd(SIDim, x + BlOCKDIM, y)], Q);
	if (iy < WINDOWDIM - 1)
		SubSplitImage[calInd(SSIDim, iy + BlOCKDIM, ix)] = convert2scale(SplitImage[calInd(SIDim, x, y + BlOCKDIM)], Q);
	if (ix < WINDOWDIM - 1 && iy < WINDOWDIM - 1)
		SubSplitImage[calInd(SSIDim, iy + BlOCKDIM, ix + BlOCKDIM)] 
			= convert2scale(SplitImage[calInd(SIDim, x + BlOCKDIM, y + BlOCKDIM)], Q);
	__syncthreads();
}

__global__ void gpuCalculateTexture(float *Texture, int TextureDim, float *SplitImage, int SIDim, float *Q, int method, int bool_mean)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	if (x >= TextureDim || y >= TextureDim)
		return;
	__shared__ int SubSplitImage[(BlOCKDIM + WINDOWDIM - 1)*(BlOCKDIM + WINDOWDIM - 1)];
	int SSIDim = BlOCKDIM + WINDOWDIM - 1;
	copyAndConvertImage(SplitImage, SIDim, SubSplitImage, SSIDim, Q, x, y, ix, iy);
	if (SplitImage[calInd(SIDim, x + (WINDOWDIM - 1) / 2, y + (WINDOWDIM - 1) / 2)] == DEFAULT)
		Texture[calInd(TextureDim, x, y)] = DEFAULT;
	else
	{				
		int jstart, jstop, istart, istop, xshift, yshift;
		int value1, value2;
		float texture[4] = { 0.0 };			
		for (int t = 0; t < 4; t++)
		{
			//X=D, Y=0 shift
			if (t == 0)
			{				
				xshift = D; yshift = 0;		
				jstart = iy; jstop = WINDOWDIM + iy; istart = ix; istop = WINDOWDIM - D + ix;
			}
			//X=D, Y=-D shift
			if (t == 1)
			{
				xshift = D; yshift = -1 * D;
				jstart = D + iy; jstop = WINDOWDIM + iy; istart = ix; istop = WINDOWDIM - D + ix;
			}
			//X=0, Y=D shift
			if (t == 2)
			{
				xshift = 0; yshift = D;
				jstart = iy; jstop = WINDOWDIM - D + iy; istart = ix; istop = WINDOWDIM + ix;
			}
			//X=D, Y=D shift
			if (t == 3)
			{
				xshift = D; yshift = D;
				jstart = iy; jstop = WINDOWDIM - D + iy; istart = ix; istop = WINDOWDIM - D + ix;
			}			
			unsigned char SDM[NG*NG] = { 0 };
			for (int j = jstart; j < jstop; j++)
			{
				for (int i = istart; i < istop; i++)
				{
					value1 = SubSplitImage[calInd(SSIDim, j, i)];
					value2 = SubSplitImage[calInd(SSIDim, j + yshift, i + xshift)];
					updateSDM(SDM, value1, value2);
					updateSDM(SDM, value2, value1);
				}
			}
			texture[t] = calTexture(SDM, method);
		}
		Texture[calInd(TextureDim, x, y)] = calAvgOrRange(texture, 4, bool_mean);
	}
	__syncthreads();
}