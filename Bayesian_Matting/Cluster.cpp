#include "Cluster.h"

Cluster::Cluster(const std::vector<std::pair<cv::Vec3f, float> > &sample_set)
{
	this->sample_set = sample_set;
}

void Cluster::Calc(void)
{
	float W = 0.0f;
	size_t elemNum   = sample_set.size();
	cv::Mat X = cv::Mat(elemNum, 3, CV_32FC1);
	cv::Mat w = cv::Mat(elemNum, 1, CV_32FC1);
	cv::Mat wpixels  = cv::Mat(elemNum, 3, CV_32FC1);
	cv::Vec3f color;

	for (size_t k = 0; k < elemNum; k++)
	{
		color = sample_set[k].first;

		X.at<float>(k, 0) = color[0];
		X.at<float>(k, 1) = color[1];
		X.at<float>(k, 2) = color[2];

		w.at<float>(k, 0) = sample_set[k].second;

		wpixels.at<float>(k, 0) = color[0] * sample_set[k].second;
		wpixels.at<float>(k, 1) = color[1] * sample_set[k].second;
		wpixels.at<float>(k, 2) = color[2] * sample_set[k].second;

		W = W + sample_set[k].second;
	}

	q = cv::Mat(1, 3, CV_32FC1);
	float s1 = 0.0f;
	float s2 = 0.0f;
	float s3 = 0.0f;
	for (size_t k = 0; k < elemNum; k++)
	{
		s1 = s1 + wpixels.at<float>(k, 0);
		s2 = s2 + wpixels.at<float>(k, 1);
		s3 = s3 + wpixels.at<float>(k, 2);
	}

	q.at<float>(0, 0) = s1 / W;
	q.at<float>(0, 1) = s2 / W;
	q.at<float>(0, 2) = s3 / W;

	cv::Mat qwork = cv::Mat(elemNum, 3, CV_32FC1);
	for (size_t k = 0; k < elemNum; k++)
	{
		qwork.at<float>(k, 0) = q.at<float>(0, 0);
		qwork.at<float>(k, 1) = q.at<float>(0, 1);
		qwork.at<float>(k, 2) = q.at<float>(0, 2);
	}

	cv::Mat twork = X - qwork;

	cv::Mat wwork = cv::Mat(elemNum, 3, CV_32FC1);
	for (size_t k = 0; k < elemNum; k++)
	{
		float w_sqrt = std::sqrtf(w.at<float>(k, 0));
		wwork.at<float>(k, 0) = w_sqrt;
		wwork.at<float>(k, 1) = w_sqrt;
		wwork.at<float>(k, 2) = w_sqrt;
	}

	cv::Mat t = cv::Mat(elemNum, 3, CV_32FC1);

	for (int r = 0; r < wwork.rows; r++)
	{
		for (int c = 0; c < wwork.cols; c++)
		{
			t.at<float>(r, c) = twork.at<float>(r, c) * wwork.at<float>(r, c);
		}
	}

	R = (t.t() * t) / W + cv::Mat::eye(3, 3, CV_32FC1) * 1e-5;

	/*cv::Mat eigval = cv::Mat(3, 1, CV_32FC1);
	cv::Mat eigvec = cv::Mat(3, 3, CV_32FC1);

	cv::SVD svd(R);
	svd.w.copyTo(eigval);
	svd.u.copyTo(eigvec);

	e = eigvec.col(0);
	lambda = eigval.at<float>(0, 0);*/

	cv::Mat eigenvalues;
	cv::Mat eigenvectors;

	cv::eigen(R, eigenvalues, eigenvectors);

	e = cv::Mat(3, 1, CV_32FC1);

	e.at<float>(0, 0) = eigenvectors.at<float>(0, 0);
	e.at<float>(1, 0) = eigenvectors.at<float>(0, 1);
	e.at<float>(2, 0) = eigenvectors.at<float>(0, 2);

	lambda = eigenvalues.at<float>(0, 0);
}