#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "getopt.h"
#include "bayesian.h"

using namespace std;

static void usage(const char* name)
{
	cout << name << ": A tool for Bayesian Matting" << endl;
	cout << "Usage1: " << name << " -s source -t trimap -c composite" << endl;
	cout << "Usage2: " << name << " --src source --trimap trimap -comp composite" << endl;
	cout << "Valid options ares:" << endl;
	cout << "    -s/--src     source image" << endl;
	cout << "    -t/--trimap  trimap image" << endl;
	cout << "    -c/--comp    composite image" << endl;
}

int main(int argc, char* argv[])
{
	const char* optstring = "s:t:c:";
	static struct option longOptions[] =
	{
		{"src", required_argument, NULL, 's'},
		{"trimap", required_argument, NULL, 't'},
		{"comp", required_argument, NULL, 'c'},
		0
	};

	if (argc < 4) {
		usage(argv[0]);
		return 1;
	}

	int c;
	int optionIndex = 0;
	const char* srcname  = "gandalf\\input-small.png";
	const char* triname  = "gandalf\\trimap.png";
	const char* compname = "gandalf\\background-small.png";
	
	while ((c = getopt_long(argc, argv, optstring, longOptions, &optionIndex)) != -1)
	{
		switch (c) {
			case 's':
				srcname = optarg;
				//cout << srcname << endl;
				break;
			case 't':
				triname = optarg;
				//cout << triname << endl;
				break;
			case 'c':
				compname = optarg;
				//cout << compname << endl;
				break;
			default:
				usage(argv[0]);
				return 1;
		}
	}

	cv::Mat src = cv::imread(srcname);
	if (src.data == NULL) {
		cerr << "Cann't open source image." << endl;
		return 1;
	}

	cv::Mat trimap = cv::imread(triname);
	if (trimap.data == NULL) {
		cerr << "Cann't open trimap image." << endl;
		return 1;
	}

	cv::Mat comp = cv::imread(compname);
	if (comp.data == NULL) {
		cerr << "Cann't open composite image." << endl;
		return 1;
	}

	cv::imshow("src", src);
	cv::imshow("trimap", trimap);
	cv::imshow("composite", comp);
	cv::waitKey(0);

	BayesianMatting matting(src, trimap);
	matting.Solve();

	cv::Mat result;
	matting.Composite(comp, &result);

	cv::imshow("result", result);
	cv::waitKey(0);

	return 0;
}