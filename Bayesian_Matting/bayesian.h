#ifndef __BAYESIAN_H__
#define __BAYESIAN_H__

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Cluster.h"

#define N              255
#define SIGMA          8.0f
#define SIGMA_C        5.0f
#define MIN_VAR        0.05f
#define MAX_ITERATION  50
#define MIN_LIKE       1e-6

class BayesianMatting
{
public:
	BayesianMatting(const cv::Mat& cImg, const cv::Mat& tmap);

	~BayesianMatting(void);

	void SetParameters(int nearest, float sigma, float sigma_c);

	void Solve(void);

	void Composite(const cv::Mat &composite, cv::Mat *result);

private:
	void Initialize(void);

	float BayesianMatting::InitAlpha(const int x, const int y);

	void CollectFgSamples(const int x, const int y, std::vector<std::pair<cv::Vec3f, float> > *sample_set);

	void CollectBgSamples(const int x, const int y, std::vector<std::pair<cv::Vec3f, float> > *sample_set);

	float max_lambda(const std::vector<Cluster> &nodes, int *idx);

	void Split(std::vector<Cluster> *nodes);

	void Cluster_OrchardBouman(
		const int x, const int y,
		const std::vector<std::pair<cv::Vec3f, float> > &sample_set,
		std::vector<Cluster> *clusters);

	void AddCamVar(std::vector<Cluster> *clusters);

	void BayesianMatting::Optimize(
		const int x, const int y, 
		const std::vector<Cluster> &fg_clusters, const std::vector<Cluster> &bg_clusters, const float alpha_init,
		cv::Vec3f *F, cv::Vec3f *B, float *a);
	
	float ComputeLikelihood(
		const int x, const int y,
		const cv::Mat &mu_Fi, const cv::Mat &invSigma_Fi,
		const cv::Mat &mu_Bj, const cv::Mat &invSigma_Bj,
		const cv::Vec3f &c_color, const cv::Vec3f &fg_color, const cv::Vec3f &bg_color, const float alpha);

	unsigned nUnknown;
	unsigned nearest;
	float sigma;
	float sigma_c;

	cv::Mat fgmask, bgmask, unmask;
	cv::Mat unsolvedmask;

	cv::Mat trimap;
	cv::Mat alphamap;

	cv::Mat colorImg, fgImg, bgImg;
};

#endif  // #ifndef __BAYESIAN_H__