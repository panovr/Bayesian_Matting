#ifndef __CLUSTER_H__
#define __CLUSTER_H__

#include <vector>
#include <opencv2/core/core.hpp>

class Cluster
{
public:
	Cluster(void) {}

	Cluster(const std::vector<std::pair<cv::Vec3f, float> > &sample_set);

	void Calc(void);

	std::vector<std::pair<cv::Vec3f, float> > sample_set;

	cv::Mat q;
	cv::Mat R;
	cv::Mat e;
	float lambda;
};

#endif __CLUSTER_H__  // #ifndef __CLUSTER_H__