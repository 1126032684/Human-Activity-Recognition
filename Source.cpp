/*
* Name : Kanav Kaul
* NetId : kxk140730
*/

//#include "stdafx.h"
#include <opencv/cv.h>  
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/highgui.h> 
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <time.h>  
#include <math.h>  
#include <ctype.h>  
#include <stdio.h>  
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "opencv2/video/tracking.hpp"
#include <math.h>
#include <time.h>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include  <opencv/ml.h>	
#include <fstream>
#include<sstream>
#include <opencv2/nonfree/nonfree.hpp>

using namespace cv;
using namespace std;



//--------Using SIFT as feature extractor and FlannBased for assigning a new point to the nearest one in the dictionary
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
Ptr<DescriptorExtractor> extractor = new SiftDescriptorExtractor();
SiftFeatureDetector detector(100);
//---dictionary size=number of cluster's centroids
int clusterSize = 1500;
TermCriteria tc(CV_TERMCRIT_ITER, 10, 0.001);
int retries = 1;
int flags = KMEANS_PP_CENTERS;
BOWKMeansTrainer bowTrainer(clusterSize, tc, retries, flags);
BOWImgDescriptorExtractor bowDE(extractor, matcher);



// various tracking parameters (in seconds)
const double MHI_DURATION = 1;
const double MAX_TIME_DELTA = 0.5;
const double MIN_TIME_DELTA = 0.05;
const double threshold_value = 25;
Mat SAMHI_10 = Mat(480, 640, CV_32FC1, Scalar(0, 0, 0));;
int main(int argc, char** argv)
{
	IplImage* motionImage = 0;
	VideoCapture cap = 0;
	fstream in("train.txt");
	for (int fnum = 1; fnum < 74; fnum++)
	{
		
		
		char fname[5];
		int lab;
		in>>lab;
		in>>fname;
		
		cap = VideoCapture(fname);
		
		Mat image_binary_prev_5, image_binary, image_binary_diff_5;
		int nframes = cap.get(CV_CAP_PROP_FRAME_COUNT); // Getiing number of frames
		for (int i = 1; i < nframes; i++)
		{
			Mat frame;
			// Retrieve a single frame from the capture
			cap.read(frame);

			cvtColor(frame, image_binary, CV_BGR2GRAY);

			int num = 5;

			if (i == 1){
				image_binary_prev_5 = image_binary.clone();
			}
			//to perform differences between frames adjacent by 5 frames.
			if (i % num == 0){
				absdiff(image_binary_prev_5, image_binary, image_binary_diff_5);
				image_binary_prev_5 = image_binary.clone();
			}

			//to create an initial samhi image of 5;
			if (i == num + 1){
				threshold(image_binary_diff_5, image_binary_diff_5, threshold_value, 255, THRESH_BINARY);
				Size framesize = image_binary_diff_5.size();
				int h = framesize.height;
				int w = framesize.width;
				SAMHI_10 = Mat(h, w, CV_32FC1, Scalar(0, 0, 0));
				updateMotionHistory(image_binary_diff_5, SAMHI_10, (double)i / nframes, MHI_DURATION);
			}
			//creating SAMHI image of 5
			if (i > num + 1 && i % num == 0){
				threshold(image_binary_diff_5, image_binary_diff_5, threshold_value, 255, THRESH_BINARY);
				updateMotionHistory(image_binary_diff_5, SAMHI_10, (double)i / nframes, MHI_DURATION);
			}

		}
		cout<<"File "<<fname<<" MHI done.."<<endl;
		
		vector<KeyPoint> keypoint;
		SAMHI_10.convertTo(SAMHI_10, CV_8UC1, 255, 0);
		detector.detect(SAMHI_10, keypoint);
		Mat features;
		extractor->compute(SAMHI_10, keypoint, features);
		bowTrainer.add(features);

	}
	
	in.close();
	vector<Mat> descriptors = bowTrainer.getDescriptors();
	int count = 0;
	for (vector<Mat>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
	{
		count += iter->rows;
	}
	
	Mat dictionary = bowTrainer.cluster();
	bowDE.setVocabulary(dictionary);
	//cout << "extracting histograms in the form of BOW for each image " << endl;
	Mat labels(0, 1, CV_32FC1);
	Mat trainingData(0, clusterSize, CV_32FC1);
	int k = 0;
	vector<KeyPoint> keypoint1;
	Mat bowDescriptor1;
	in.open("train.txt");
	for (int fnum = 1; fnum < 74; fnum++)
	{
		
		char fname[5];
		float l;
		in>>l;
		in>>fname;
		cap = VideoCapture(fname);
		//IplImage*  frame;
		Mat image_binary_prev_5, image_binary, image_binary_diff_5;
		int nframes = cap.get(CV_CAP_PROP_FRAME_COUNT);
		for (int i = 1; i < nframes; i++)
		{
			Mat frame;
			// Retrieve a single frame from the capture
			cap.read(frame);

			cvtColor(frame, image_binary, CV_BGR2GRAY);


			int num = 5;

			if (i == 1){
				image_binary_prev_5 = image_binary.clone();
			}
			//to perform differences between frames adjacent by 5 frames.
			if (i % num == 0){
				absdiff(image_binary_prev_5, image_binary, image_binary_diff_5);
				image_binary_prev_5 = image_binary.clone();
			}

			//to create an initial samhi image of 5;
			if (i == num + 1){
				threshold(image_binary_diff_5, image_binary_diff_5, threshold_value, 255, THRESH_BINARY);
				Size framesize = image_binary_diff_5.size();
				int h = framesize.height;
				int w = framesize.width;
				SAMHI_10 = Mat(h, w, CV_32FC1, Scalar(0, 0, 0));
				updateMotionHistory(image_binary_diff_5, SAMHI_10, (double)i / nframes, MHI_DURATION);
			}
			//creating SAMHI image of 5
			if (i > num + 1 && i % num == 0){
				threshold(image_binary_diff_5, image_binary_diff_5, threshold_value, 255, THRESH_BINARY);
				updateMotionHistory(image_binary_diff_5, SAMHI_10, (double)i / nframes, MHI_DURATION);
			}

		}
		//extracting histogram in the form of bow for each image 
		SAMHI_10.convertTo(SAMHI_10, CV_8UC1, 255, 0);
		detector.detect(SAMHI_10, keypoint1);


		bowDE.compute(SAMHI_10, keypoint1, bowDescriptor1);
		labels.push_back(l);
		trainingData.push_back(bowDescriptor1);



	}

	in.close();
	//Setting up SVM parameters
	CvSVMParams params;
	params.kernel_type = CvSVM::RBF;
	params.svm_type = CvSVM::C_SVC;
	params.gamma = 0.50625000000000009;
	params.C = 312.50000000000000;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 0.000001);
	CvSVM svm_classif;



	printf("%s\n", "Training SVM classifier");

	bool res = svm_classif.train(trainingData, labels, cv::Mat(), cv::Mat(), params);

	Mat groundTruth(0, 1, CV_32FC1);
	Mat evalData(0, clusterSize, CV_32FC1);
	k = 0;
	vector<KeyPoint> keypoint2;
	Mat bowDescriptor2;


	Mat results(0, 1, CV_32FC1);
	in.open("testdemonewline.txt");
	/// Testing
	for (int fnum = 1; fnum <=75; fnum++)
	{
	
		
		string fname;
		float l;
		in>>l;
		in>>fname;
		cap = VideoCapture(fname);
		
		Mat image_binary_prev_5, image_binary, image_binary_diff_5;
		int nframes = cap.get(CV_CAP_PROP_FRAME_COUNT);
		for (int i = 1; i < nframes; i++)
		{

			Mat frame;
			// Retrieve a single frame from the capture
			cap.read(frame);

			cvtColor(frame, image_binary, CV_BGR2GRAY);


			int num = 5;

			if (i == 1){
				image_binary_prev_5 = image_binary.clone();
			}
			//to perform differences between frames adjacent by 5 frames.
			if (i % num == 0){
				absdiff(image_binary_prev_5, image_binary, image_binary_diff_5);
				image_binary_prev_5 = image_binary.clone();
			}

			//to create an initial samhi image of 5;
			if (i == num + 1){
				threshold(image_binary_diff_5, image_binary_diff_5, threshold_value, 255, THRESH_BINARY);
				Size framesize = image_binary_diff_5.size();
				int h = framesize.height;
				int w = framesize.width;
				SAMHI_10 = Mat(h, w, CV_32FC1, Scalar(0, 0, 0));
				updateMotionHistory(image_binary_diff_5, SAMHI_10, (double)i / nframes, MHI_DURATION);
			}
			//creating SAMHI image of 5
			if (i > num + 1 && i % num == 0){
				threshold(image_binary_diff_5, image_binary_diff_5, threshold_value, 255, THRESH_BINARY);
				updateMotionHistory(image_binary_diff_5, SAMHI_10, (double)i / nframes, MHI_DURATION);
			}

		}

		
		SAMHI_10.convertTo(SAMHI_10, CV_8UC1, 255, 0);
		detector.detect(SAMHI_10, keypoint2);
		bowDE.compute(SAMHI_10, keypoint2, bowDescriptor2);

		evalData.push_back(bowDescriptor2);
		groundTruth.push_back(l);

		float result = svm_classif.predict(bowDescriptor2);

		cout << "File "<<fnum<<" tested:: " << result << endl;
		results.push_back(result);
	}
	in.close();


	//calculate the number of unmatched classes 
	double err = (double)countNonZero(groundTruth - results) / evalData.rows;
	printf("%s%f", "Accuracy : ", (1-err)*100);
	printf("%");

	return 0;
}