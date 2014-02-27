#include <iostream>
#include <climits>

#include "bayesian.h"

BayesianMatting::BayesianMatting(const cv::Mat& cImg, const cv::Mat& tmap)
{
	CV_Assert(cImg.size() == tmap.size());

	cImg.convertTo(colorImg, CV_32F, 1.0f / 255.0f);

	// Convert the trimap into a single channel image
	if (tmap.channels() == 3)
	{
		cv::cvtColor(tmap, trimap, CV_RGB2GRAY);
	}
	else if (tmap.channels() == 1)
	{
		this->trimap = tmap.clone();
	}

	Initialize();

	SetParameters(N, SIGMA, SIGMA_C);
}

BayesianMatting::~BayesianMatting(void)
{
}

void BayesianMatting::Initialize(void)
{
	nUnknown = 0;

	cv::Size imgSize = trimap.size();

	this->fgImg			= cv::Mat(imgSize, CV_32FC3, cv::Scalar(0, 0, 0));
	this->bgImg			= cv::Mat(imgSize, CV_32FC3, cv::Scalar(0, 0, 0));
	this->fgmask		= cv::Mat(imgSize, CV_8UC1, cv::Scalar(0));
	this->bgmask		= cv::Mat(imgSize, CV_8UC1, cv::Scalar(0));
	this->unmask		= cv::Mat(imgSize, CV_8UC1, cv::Scalar(0));
	this->alphamap		= cv::Mat(imgSize, CV_32FC1, cv::Scalar(0));

	for (int y = 0; y < imgSize.height; y++)
	{
		for (int x = 0; x < imgSize.width; x++)
		{
			uchar v = trimap.at<uchar>(y, x);
			if (v == 0)
			{
				bgmask.at<uchar>(y, x) = 255;
			}
			else if (v == 255)
			{
				fgmask.at<uchar>(y, x) = 255;
			}
			else
			{
				unmask.at<uchar>(y, x) = 255;
			}
		}
	}

	this->colorImg.copyTo(fgImg, fgmask);
	this->colorImg.copyTo(bgImg, bgmask);
	this->alphamap.setTo(cv::Scalar(0), bgmask);
	this->alphamap.setTo(cv::Scalar(1), fgmask);
	this->alphamap.setTo(cv::Scalar(-1.0f), unmask);
	this->unsolvedmask = unmask.clone();

	for (int r = 0; r < unmask.rows; r++)
	{
		for (int c = 0; c < unmask.cols; c++)
		{
			if (unmask.at<uchar>(r, c) == 255)
			{
				nUnknown = nUnknown + 1;
			}
		}
	}

	/*std::cout << "nUnknown = " << nUnknown << std::endl;
	cv::imshow("fgmask", fgmask);
	cv::imshow("bgmask", bgmask);
	cv::imshow("unmask", unmask);
	cv::imshow("fgImg", fgImg);
	cv::imshow("bgImg", bgImg);
	cv::imshow("alpha", alphamap);
	cv::waitKey(0);*/
}

void BayesianMatting::SetParameters(int nearest, float sigma, float sigma_c)
{
	this->nearest = nearest;
	this->sigma = sigma;
	this->sigma_c = sigma_c;
}

float BayesianMatting::InitAlpha(const int x, const int y)
{
	unsigned alpha_num = 0;
    float alpha_init = 0;
    int dist = 1;
    
    while (alpha_num < nearest)
    {
        if (y - dist >= 0)
        {
            for (int z = std::max(0, x - dist); z <= std::min(alphamap.cols - 1, x + dist); z++)
            {
                // We know this pixel belongs to the foreground
                if (fgmask.at<uchar>(y - dist, z) != 0)
                {
					alpha_init = alpha_init + alphamap.at<float>(y - dist, z);
					alpha_num = alpha_num + 1;
                    if (alpha_num == nearest)
					{
                        goto DONE;
					}
                }
				else if (unmask.at<uchar>(y - dist, z) != 0 && unsolvedmask.at<uchar>(y - dist, z) == 0)
				{
                    alpha_init = alpha_init + alphamap.at<float>(y - dist, z);
					alpha_num = alpha_num + 1;
                    if (alpha_num == nearest)
					{
                        goto DONE;
					}
                }
				else if (bgmask.at<uchar>(y - dist, z) != 0)
				{
					alpha_init = alpha_init + alphamap.at<float>(y - dist, z);
					alpha_num = alpha_num + 1;
                    if (alpha_num == nearest)
					{
                        goto DONE;
					}
				}
            }
        }
        
        if (y + dist < alphamap.rows)
        {
            for (int z = std::max(0, x - dist); z <= std::min(alphamap.cols - 1, x + dist); z++)
            {
                if (fgmask.at<uchar>(y + dist, z) != 0)
                {
					alpha_init = alpha_init + alphamap.at<float>(y + dist, z);
					alpha_num = alpha_num + 1;
                    if (alpha_num == nearest)
					{
                        goto DONE;
					}
                }
				else if (unmask.at<uchar>(y + dist, z) != 0 && unsolvedmask.at<uchar>(y + dist, z) == 0)
                {
					alpha_init = alpha_init + alphamap.at<float>(y + dist, z);
					alpha_num = alpha_num + 1;
                    if (alpha_num == nearest)
					{
                        goto DONE;
					}
                }
				else if (bgmask.at<uchar>(y + dist, z) != 0)
				{
					alpha_init = alpha_init + alphamap.at<float>(y + dist, z);
					alpha_num = alpha_num + 1;
                    if (alpha_num == nearest)
					{
                        goto DONE;
					}
				}
            }
        }
        
        if (x - dist >= 0)
        {
            for(int z = std::max(0, y - dist + 1); z <= std::min(alphamap.rows - 1, y + dist - 1); z++)
            {
                if(fgmask.at<uchar>(z, x - dist) != 0)
                {
					alpha_init = alpha_init + alphamap.at<float>(z, x - dist);
					alpha_num = alpha_num + 1;
                    if (alpha_num == nearest)
					{
                        goto DONE;
					}
                }
				else if (unmask.at<uchar>(z, x - dist) != 0 && unsolvedmask.at<uchar>(z, x - dist) == 0)
                {
					alpha_init = alpha_init + alphamap.at<float>(z, x - dist);
					alpha_num = alpha_num + 1;
                    if (alpha_num == nearest)
					{
                        goto DONE;
					}
                }
				else if (bgmask.at<uchar>(z, x - dist) != 0)
				{
					alpha_init = alpha_init + alphamap.at<float>(z, x - dist);
					alpha_num = alpha_num + 1;
                    if (alpha_num == nearest)
					{
                        goto DONE;
					}
				}
            }
        }
        
        if (x + dist < alphamap.cols)
        {
            for (int z = std::max(0, y - dist + 1); z <= std::min(alphamap.rows - 1, y + dist - 1); z++)
            {
                if (fgmask.at<uchar>(z, x + dist) != 0)
                {
					alpha_init = alpha_init + alphamap.at<float>(z, x + dist);
					alpha_num = alpha_num + 1;
                    if (alpha_num == nearest)
					{
                        goto DONE;
					}
                }
                else if (unmask.at<uchar>(z, x + dist) != 0 && unsolvedmask.at<uchar>(z, x + dist) == 0)
                {
					alpha_init = alpha_init + alphamap.at<float>(z, x + dist);
					alpha_num = alpha_num + 1;
                    if (alpha_num == nearest)
					{
                        goto DONE;
					}
                }
				else if (bgmask.at<uchar>(z, x + dist) != 0)
				{
					alpha_init = alpha_init + alphamap.at<float>(z, x + dist);
					alpha_num = alpha_num + 1;
                    if (alpha_num == nearest)
					{
                        goto DONE;
					}
				}
			}
		}
		
		dist = dist + 1;
    }
	
DONE:
	return (alpha_init / alpha_num);
}

void BayesianMatting::CollectFgSamples(const int x, const int y, std::vector<std::pair<cv::Vec3f, float> > *sample_set)
{
    sample_set->clear();
    
    std::pair<cv::Vec3f, float> sample;
    float dist_weight;
    float inv_2sigma_square = 1.0f / (2.0f * this->sigma * this->sigma);
    int dist = 1;
    
    while (sample_set->size() < nearest)
    {
        if (y - dist >= 0)
        {
            for (int z = std::max(0, x - dist); z <= std::min(fgImg.cols - 1, x + dist); z++)
            {
                dist_weight = std::expf(-(dist * dist + (z - x) * (z - x)) * inv_2sigma_square);
                
                if (fgmask.at<uchar>(y - dist, z) != 0)
                {
					sample.first = fgImg.at<cv::Vec3f>(y - dist, z);
                    sample.second = dist_weight;
                    sample_set->push_back(sample);

                    if (sample_set->size() == nearest)
					{
                        goto DONE;
					}
                }
				else if (unmask.at<uchar>(y - dist, z) != 0 && unsolvedmask.at<uchar>(y - dist, z) == 0)
				{
					sample.first = fgImg.at<cv::Vec3f>(y - dist, z);
                    float alpha = alphamap.at<float>(y - dist, z);
                    sample.second = dist_weight * alpha * alpha;
                    sample_set->push_back(sample);

                    if (sample_set->size() == nearest)
					{
                        goto DONE;
					}
                }
            }
        }
        
        if (y + dist < fgImg.rows)
        {
            for (int z = std::max(0, x - dist); z <= std::min(fgImg.cols - 1, x + dist); z++)
            {
                dist_weight = std::expf(-(dist * dist + (z - x) * (z - x)) * inv_2sigma_square);
                
                if (fgmask.at<uchar>(y + dist, z) != 0)
                {
					sample.first = fgImg.at<cv::Vec3f>(y + dist, z);
                    sample.second = dist_weight;
                    sample_set->push_back(sample);

                    if (sample_set->size() == nearest)
					{
                        goto DONE;
					}
                }
				else if (unmask.at<uchar>(y + dist, z) != 0 && unsolvedmask.at<uchar>(y + dist, z) == 0)
                {
					sample.first = fgImg.at<cv::Vec3f>(y + dist, z);                
                    float alpha = alphamap.at<float>(y + dist, z);
                    sample.second = dist_weight * alpha * alpha;
                    sample_set->push_back(sample);

                    if (sample_set->size() == nearest)
					{
                        goto DONE;
					}
                }
            }
        }
        
        if (x - dist >= 0)
        {
            for(int z = std::max(0, y - dist + 1); z <= std::min(fgImg.rows - 1, y + dist - 1); z++)
            {
                dist_weight = std::expf(-(dist * dist + (z - y) * (z - y)) * inv_2sigma_square);

                if(fgmask.at<uchar>(z, x - dist) != 0)
                {
					sample.first = fgImg.at<cv::Vec3f>(z, x - dist);
                    sample.second = dist_weight;
                    sample_set->push_back(sample);

                    if (sample_set->size() == nearest)
					{
                        goto DONE;
					}
                }
				else if (unmask.at<uchar>(z, x - dist) != 0 && unsolvedmask.at<uchar>(z, x - dist) == 0)
                {
					sample.first = fgImg.at<cv::Vec3f>(z, x - dist);                   
                    float alpha = alphamap.at<float>(z, x - dist);
                    sample.second = dist_weight * alpha * alpha;              
                    sample_set->push_back(sample);

                    if (sample_set->size() == nearest)
					{
                        goto DONE;
					}
                }
            }
        }
        
        if (x + dist < fgImg.cols)
        {
            for (int z = std::max(0, y - dist + 1); z <= std::min(fgImg.rows - 1, y + dist - 1); z++)
            {
                dist_weight = std::expf(-(dist * dist + (y - z) * (y - z)) * inv_2sigma_square);
                
                if (fgmask.at<uchar>(z, x + dist) != 0)
                {
					sample.first = fgImg.at<cv::Vec3f>(z, x + dist);
                    sample.second = dist_weight;  
                    sample_set->push_back(sample);

                    if (sample_set->size() == nearest)
					{
                        goto DONE;
					}
                }
                else if (unmask.at<uchar>(z, x + dist) != 0 && unsolvedmask.at<uchar>(z, x + dist) == 0)
                {
					sample.first = fgImg.at<cv::Vec3f>(z, x + dist);
                    float alpha = alphamap.at<float>(z, x + dist);
                    sample.second = dist_weight * alpha * alpha;
                    sample_set->push_back(sample);

                    if (sample_set->size() == nearest)
					{
                        goto DONE;
					}
                }
            }
        }
		
		dist = dist + 1;
    }
	
DONE:
	CV_Assert(sample_set->size() == nearest);
}

void BayesianMatting::CollectBgSamples(const int x, const int y, std::vector<std::pair<cv::Vec3f, float> > *sample_set)
{
    sample_set->clear();
    
    std::pair<cv::Vec3f, float> sample;
    float dist_weight;
    float inv_2sigma_square = 1.0f / (2.0f * this->sigma * this->sigma);
    int dist = 1;
    
    while (sample_set->size() < nearest)
    {
        if (y - dist >= 0)
        {
            for (int z = std::max(0, x - dist); z <= std::min(bgImg.cols - 1, x + dist); z++)
            {
                dist_weight = std::expf(-(dist * dist + (z - x) * (z - x)) * inv_2sigma_square);
                
                if (bgmask.at<uchar>(y - dist, z) != 0)
                {
					sample.first = bgImg.at<cv::Vec3f>(y - dist, z);
                    sample.second = dist_weight;
                    sample_set->push_back(sample);

                    if (sample_set->size() == nearest)
					{
                        goto DONE;
					}
                }
				else if (unmask.at<uchar>(y - dist, z) != 0 && unsolvedmask.at<uchar>(y - dist, z) == 0)
				{
					sample.first = bgImg.at<cv::Vec3f>(y - dist, z);
                    float alpha = alphamap.at<float>(y - dist, z);
                    sample.second = dist_weight * (1 - alpha) * (1 - alpha);
                    sample_set->push_back(sample);

                    if (sample_set->size() == nearest)
					{
                        goto DONE;
					}
                }
            }
        }
        
        if (y + dist < bgImg.rows)
        {
            for (int z = std::max(0, x - dist); z <= std::min(bgImg.cols - 1, x + dist); z++)
            {
                dist_weight = std::expf(-(dist * dist + (z - x) * (z - x)) * inv_2sigma_square);
                
                if (bgmask.at<uchar>(y + dist, z) != 0)
                {
					sample.first = bgImg.at<cv::Vec3f>(y + dist, z);
                    sample.second = dist_weight;
                    sample_set->push_back(sample);

                    if (sample_set->size() == nearest)
					{
                        goto DONE;
					}
                }
				else if (unmask.at<uchar>(y + dist, z) != 0 && unsolvedmask.at<uchar>(y + dist, z) == 0)
                {
					sample.first = bgImg.at<cv::Vec3f>(y + dist, z);                
                    float alpha = alphamap.at<float>(y + dist, z);
                    sample.second = dist_weight * (1 - alpha) * (1 - alpha);
                    sample_set->push_back(sample);

                    if (sample_set->size() == nearest)
					{
                        goto DONE;
					}
                }
            }
        }
        
        if (x - dist >= 0)
        {
            for(int z = std::max(0, y - dist + 1); z <= std::min(bgImg.rows - 1, y + dist - 1); z++)
            {
                dist_weight = std::expf(-(dist * dist + (z - y) * (z - y)) * inv_2sigma_square);

                if (bgmask.at<uchar>(z, x - dist) != 0)
                {
					sample.first = bgImg.at<cv::Vec3f>(z, x - dist);
                    sample.second = dist_weight;
                    sample_set->push_back(sample);

                    if (sample_set->size() == nearest)
					{
                        goto DONE;
					}
                }
				else if (unmask.at<uchar>(z, x - dist) != 0 && unsolvedmask.at<uchar>(z, x - dist) == 0)
                {
					sample.first = bgImg.at<cv::Vec3f>(z, x - dist);                   
                    float alpha = alphamap.at<float>(z, x - dist);
                    sample.second = dist_weight * (1 - alpha) * (1 - alpha);              
                    sample_set->push_back(sample);

                    if (sample_set->size() == nearest)
					{
                        goto DONE;
					}
                }
            }
        }
        
        if (x + dist < bgImg.cols)
        {
            for (int z = std::max(0, y - dist + 1); z <= std::min(bgImg.rows - 1, y + dist - 1); z++)
            {
                dist_weight = std::expf(-(dist * dist + (y - z) * (y - z)) * inv_2sigma_square);
                
                if (bgmask.at<uchar>(z, x + dist) != 0)
                {
					sample.first = bgImg.at<cv::Vec3f>(z, x + dist);
                    sample.second = dist_weight;  
                    sample_set->push_back(sample);

                    if (sample_set->size() == nearest)
					{
                        goto DONE;
					}
                }
                else if (unmask.at<uchar>(z, x + dist) != 0 && unsolvedmask.at<uchar>(z, x + dist) == 0)
                {
					sample.first = bgImg.at<cv::Vec3f>(z, x + dist);
                    float alpha = alphamap.at<float>(z, x + dist);
                    sample.second = dist_weight * (1 - alpha) * (1 - alpha);
                    sample_set->push_back(sample);

                    if (sample_set->size() == nearest)
					{
                        goto DONE;
					}
                }
            }
        }
		
		dist = dist + 1;
    }
	
DONE:
	CV_Assert(sample_set->size() == nearest);
}

void BayesianMatting::Solve(void)
{
	std::cout << std::endl;
	std::cout << "===================================================================" << std::endl;
	std::cout << "Bayessian Matting application written by Yili Zhao, Copyright 2014." << std::endl;
	std::cout << "===================================================================" << std::endl;
	std::cout << std::endl << "There are total " << nUnknown << " pixels need to be solved" << std::endl;

	cv::Mat element    = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::Mat unkreg     = unsolvedmask.clone();
	cv::Mat unkreg_not;
	cv::Mat unkpixels;
	std::vector<cv::Point> toSolveList;

	std::vector<std::pair<cv::Vec3f, float> > fg_set;
	std::vector<std::pair<cv::Vec3f, float> > bg_set;

	std::vector<Cluster> fg_clusters;
	std::vector<Cluster> bg_clusters;

	float alpha_init;

	cv::Vec3f f;
	cv::Vec3f b;
	float alpha;

	unsigned loop = 1;
	unsigned n = 0;

	while (n < nUnknown)
	{
		std::cout << std::endl << "Starting iteration: " << loop << std::endl;

		cv::erode(unkreg, unkreg, element);
		cv::bitwise_not(unkreg, unkreg_not);
		cv::bitwise_and(unkreg_not, unsolvedmask, unkpixels);

		toSolveList.clear();

		for (int r = 0; r < unkpixels.rows; r++)
		{
			for (int c = 0; c < unkpixels.cols; c++)
			{
				if (unkpixels.at<uchar>(r, c) == 255)
				{
					toSolveList.push_back(cv::Point(c, r));
				}
			}
		}

		std::cout << "Find " << toSolveList.size() << " pixels to be solved" << std::endl;

		/*cv::imshow("unmask", unmask);
		cv::imshow("unkreg", unkreg);
		cv::imshow("not", unkreg_not);
		cv::imshow("unkpixels", unkpixels);
		cv::waitKey(0);*/

		for (size_t k = 0; k < toSolveList.size(); k++)
		{
			int x = toSolveList[k].x;
			int y = toSolveList[k].y;

			alpha_init = InitAlpha(x, y);
			CollectFgSamples(x, y, &fg_set);
			CollectBgSamples(x, y, &bg_set);

			Cluster_OrchardBouman(x, y, fg_set, &fg_clusters);
			Cluster_OrchardBouman(x, y, bg_set, &bg_clusters);

			AddCamVar(&fg_clusters);
			AddCamVar(&bg_clusters);

			Optimize(x, y, fg_clusters, bg_clusters, alpha_init, &f, &b, &alpha);
			
			// Update foreground, background and alpha values
			fgImg.at<cv::Vec3f>(y, x) = f;
			bgImg.at<cv::Vec3f>(y, x) = b;
			alphamap.at<float>(y, x)  = alpha;

			// Remove solved pixels from unsolvedmask to indicate it was solved
			unsolvedmask.at<uchar>(y, x) = 0;
		}

		n = n + toSolveList.size();
		loop = loop + 1;
	}

	std::cout << std::endl;
	std::cout << "Total " << n << " pixels have been solved" << std::endl;

	cv::imshow("alphamap", alphamap);
	cv::waitKey(0);
}

void BayesianMatting::Optimize(
	const int x, const int y, 
	const std::vector<Cluster> &fg_clusters, const std::vector<Cluster> &bg_clusters, const float alpha_init,
	cv::Vec3f *F, cv::Vec3f *B, float *a)
{
	float alpha1, alpha2;
	float alpha, max_alpha;
	float like, lastLike, maxLike;
	cv::Vec3f c_color, fg_color, bg_color;
	cv::Vec3f max_fg_color, max_bg_color;

	c_color  = colorImg.at<cv::Vec3f>(y, x);

	cv::Mat A = cv::Mat(6, 6, CV_32FC1, cv::Scalar(0));
	cv::Mat X = cv::Mat(6, 1, CV_32FC1, cv::Scalar(0));
	cv::Mat b = cv::Mat(6, 1, CV_32FC1, cv::Scalar(0));
	cv::Mat I = cv::Mat::eye(3, 3, CV_32FC1);

	cv::Mat work_3x3 = cv::Mat(3, 3, CV_32FC1, cv::Scalar(0));
	cv::Mat work_3x1 = cv::Mat(3, 1, CV_32FC1, cv::Scalar(0));

	cv::Mat work_c = cv::Mat(3, 1, CV_32FC1);
	cv::Mat work_f = cv::Mat(3, 1, CV_32FC1);
	cv::Mat work_b = cv::Mat(3, 1, CV_32FC1);
	cv::Mat work_1x1_n = cv::Mat(1, 1, CV_32FC1);
	cv::Mat work_1x1_d = cv::Mat(1, 1, CV_32FC1);

	max_alpha = alpha_init;
	maxLike  = -FLT_MAX;

	cv::Mat mu_Fi;
	cv::Mat invSigma_Fi;

	cv::Mat mu_Bj;
	cv::Mat invSigma_Bj;

	int iter;

	for (size_t i = 0; i < fg_clusters.size(); i++)
	{
		mu_Fi        = fg_clusters[i].q.t();
		invSigma_Fi  = fg_clusters[i].R.inv();

		for (size_t j = 0; j < bg_clusters.size(); j++)
		{
			mu_Bj        = bg_clusters[j].q.t();
			invSigma_Bj  = bg_clusters[j].R.inv();

			alpha = alpha_init;
			lastLike = -FLT_MAX;
			iter = 1;

			while (1)
			{
				// Solve for foreground and background
				float inv_sigmac_square = 1.0f / (sigma_c * sigma_c);

				work_3x3 = I * (alpha * alpha * inv_sigmac_square);
				work_3x3 = work_3x3 + invSigma_Fi;

				for (int h = 0; h < 3; h++)
				{
					for (int k = 0; k < 3; k++)
					{
						A.at<float>(h, k) = work_3x3.at<float>(h, k);
					}
				}

				work_3x3 = I * (alpha * (1 - alpha) * inv_sigmac_square);
				for (int h = 0; h < 3; h++)
				{
					for (int k = 0; k < 3; k++)
					{
						A.at<float>(h, 3 + k) = work_3x3.at<float>(h, k);
						A.at<float>(3 + h, k) = work_3x3.at<float>(h, k);
					}
				}

				work_3x3 = I * (1 - alpha) * (1 - alpha) * inv_sigmac_square;
				work_3x3 = work_3x3 + invSigma_Bj;
				for (int h = 0; h < 3; h++)
				{
					for (int k = 0; k < 3; k++)
					{
						A.at<float>(3 + h, 3 + k) = work_3x3.at<float>(h, k);
					}
				}

				work_3x1 = invSigma_Fi * mu_Fi;
				for (int h = 0; h < 3; h++)
				{
					b.at<float>(h, 0) = work_3x1.at<float>(h, 0) + c_color[h] * alpha * inv_sigmac_square;
				}
				
				work_3x1 = invSigma_Bj * mu_Bj;
				for (int h = 0; h < 3; h++)
				{
					b.at<float>(3 + h, 0) = work_3x1.at<float>(h, 0) + c_color[h] * (1 - alpha) * inv_sigmac_square;
				}

				cv::solve(A, b, X);

				// foreground color
				fg_color[0] = static_cast<float>(std::max(0.0, std::min(1.0, (double)X.at<float>(0, 0))));
				fg_color[1] = static_cast<float>(std::max(0.0, std::min(1.0, (double)X.at<float>(1, 0))));
				fg_color[2] = static_cast<float>(std::max(0.0, std::min(1.0, (double)X.at<float>(2, 0))));

				// background color
				bg_color[0] = static_cast<float>(std::max(0.0, std::min(1.0, (double)X.at<float>(3, 0))));
				bg_color[1] = static_cast<float>(std::max(0.0, std::min(1.0, (double)X.at<float>(4, 0))));
				bg_color[2] = static_cast<float>(std::max(0.0, std::min(1.0, (double)X.at<float>(5, 0))));

				// Solve for alpha
				for (int h = 0; h < 3; h++)
				{
					work_c.at<float>(h, 0) = c_color[h];
					work_f.at<float>(h, 0) = fg_color[h];
					work_b.at<float>(h, 0) = bg_color[h];
				}

				work_3x1 = work_c - work_b;
				work_1x1_n = work_3x1.t() * (work_f - work_b);

				work_3x1 = work_f - work_b;
				work_1x1_d = work_3x1.t() * work_3x1;

				alpha1 = work_1x1_n.at<float>(0, 0);
				alpha2 = work_1x1_d.at<float>(0, 0);

				alpha = static_cast<float>(std::max(0.0, std::min(1.0, static_cast<double>(alpha1) / static_cast<double>(alpha2))));

				like = ComputeLikelihood(x, y, mu_Fi, invSigma_Fi, mu_Bj, invSigma_Bj, c_color, fg_color, bg_color, alpha);
				
				if (iter >= MAX_ITERATION || std::fabs(like - lastLike) <= MIN_LIKE)
				{
					break;
				}

				lastLike = like;
				iter = iter + 1;
			}
				
			if (like > maxLike)
			{
				maxLike = like;
				max_fg_color = fg_color;
				max_bg_color = bg_color;
				max_alpha = alpha;
			}
		}
	}

	(*F) = max_fg_color;
	(*B) = max_bg_color;
	(*a) = max_alpha;
}

void BayesianMatting::Composite(const cv::Mat &composite, cv::Mat *result)
{
	cv::Mat coImg;
	cv::Vec3f f, b;
	float alpha;

	composite.convertTo(coImg, CV_32F, 1.0f / 255.0f);
	
	result->create(composite.size(), CV_32FC3);

	for (int y = 0; y < result->rows; y++)
	{
		for (int x = 0; x < result->cols; x++)
		{
			f = fgImg.at<cv::Vec3f>(y, x);
			b = coImg.at<cv::Vec3f>(y, x);
			alpha = alphamap.at<float>(y, x);
			result->at<cv::Vec3f>(y, x)[0] = static_cast<float>(std::max(0.0, std::min(1.0, f[0] * alpha + b[0] * (1.0 - alpha))));
			result->at<cv::Vec3f>(y, x)[1] = static_cast<float>(std::max(0.0, std::min(1.0, f[1] * alpha + b[1] * (1.0 - alpha))));
			result->at<cv::Vec3f>(y, x)[2] = static_cast<float>(std::max(0.0, std::min(1.0, f[2] * alpha + b[2] * (1.0 - alpha))));
		}
	}
}

float BayesianMatting::ComputeLikelihood(
	const int x, const int y, 
	const cv::Mat &mu_Fi, const cv::Mat &invSigma_Fi,
	const cv::Mat &mu_Bj, const cv::Mat &invSigma_Bj,
	const cv::Vec3f &c_color, const cv::Vec3f &fg_color, const cv::Vec3f &bg_color, const float alpha)
{
	float L_C, L_F, L_B;
	float inv_sigmac_square = 1.0f / (sigma_c * sigma_c);

	cv::Mat work3x1 = cv::Mat(3, 1, CV_32FC1);
	cv::Mat work1x1 = cv::Mat(1, 1, CV_32FC1);

	cv::Mat F = cv::Mat(3, 1, CV_32FC1);
	cv::Mat B = cv::Mat(3, 1, CV_32FC1);
	cv::Mat C  = cv::Mat(3, 1, CV_32FC1);

	F.at<float>(0, 0) = fg_color[0];
	F.at<float>(1, 0) = fg_color[1];
	F.at<float>(2, 0) = fg_color[2];

	B.at<float>(0, 0) = bg_color[0];
	B.at<float>(1, 0) = bg_color[1];
	B.at<float>(2, 0) = bg_color[2];

	C.at<float>(0, 0) = c_color[0];
	C.at<float>(1, 0) = c_color[1];
	C.at<float>(2, 0) = c_color[2];

	work3x1 = F - mu_Fi;
	work1x1 = work3x1.t() * invSigma_Fi * work3x1;

	L_F = -1.0f * work1x1.at<float>(0, 0) / 2.0f;

	work3x1 = B - mu_Bj;
	work1x1 = work3x1.t() * invSigma_Bj * work3x1;

	L_B = -1.0f * work1x1.at<float>(0, 0) / 2.0f;

	work3x1 = C - (F * alpha) - (B * (1.0f - alpha));
	work1x1 = work3x1.t() * work3x1;

	L_C = -1.0f * work1x1.at<float>(0, 0) * inv_sigmac_square;

	return L_F + L_B + L_C;
}

float BayesianMatting::max_lambda(const std::vector<Cluster> &nodes, int *idx)
{
	CV_Assert(nodes.size() != 0);

	float max = nodes[0].lambda;
	(*idx) = 0;

	for (size_t k = 1; k < nodes.size(); k++)
	{
		if (nodes[k].lambda > max)
		{
			max = nodes[k].lambda;
			(*idx) = k;
		}
	}

	return max;
}

void BayesianMatting::Split(std::vector<Cluster> *nodes)
{
	int idx;
	max_lambda((*nodes), &idx);

	Cluster Ci = (*nodes)[idx];
	Cluster Ca;
	Cluster Cb;

	cv::Mat bo = Ci.q * Ci.e;
	double boundary = bo.at<float>(0, 0);

	cv::Mat cur_color = cv::Mat(3, 1, CV_32FC1);

	for (size_t i = 0; i < Ci.sample_set.size(); i++)
	{
		for (int j = 0; j < 3; j++)
		{
			cur_color.at<float>(j, 0) = Ci.sample_set[i].first[j];
		}

		if (cur_color.dot(Ci.e) <= boundary)
		{	
			Ca.sample_set.push_back(Ci.sample_set[i]);
		}
		else
		{
			Cb.sample_set.push_back(Ci.sample_set[i]);
		}
	}

	Ca.Calc();
	Cb.Calc();

	nodes->erase(nodes->begin() + idx);
	nodes->push_back(Ca);
	nodes->push_back(Cb);
}

void BayesianMatting::Cluster_OrchardBouman(
	const int x, const int y,
	const std::vector<std::pair<cv::Vec3f, float> > &sample_set,
	std::vector<Cluster> *Clusters)
{
	Clusters->clear();

	Cluster C1(sample_set);
	C1.Calc();
	Clusters->push_back(C1);

	int idx;
	while (max_lambda((*Clusters), &idx) > MIN_VAR)
	{
		Split(Clusters);
	}
}

void BayesianMatting::AddCamVar(std::vector<Cluster> *clusters)
{
	float sigma_c_square = sigma_c * sigma_c;
	cv::Mat diag;

	for (size_t k = 0; k < clusters->size(); k++)
	{
		diag = cv::Mat::zeros(3, 3, CV_32FC1);

		cv::SVD svd((*clusters)[k].R);

		diag.at<float>(0, 0) = svd.w.at<float>(0, 0) + sigma_c_square;
		diag.at<float>(1, 1) = svd.w.at<float>(1, 0) + sigma_c_square;
		diag.at<float>(2, 2) = svd.w.at<float>(2, 0) + sigma_c_square;
		
		(*clusters)[k].R = svd.u * diag * svd.vt;
	}
}