// ExpressionSVM.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include<iostream>
#include<sstream>
#include<String>
using namespace std;

#include"opencv2/core.hpp"
#include"opencv2/highgui.hpp"
#include"opencv2/imgproc.hpp"
#include"opencv2/ml.hpp"
using namespace cv;

int main(int argc, const char** argv)
{
	cout << "OpenCV Training SVM Emotion Recognition\n";
    cout << "\n";

    const char* path_Happy;
    const char* path_Sad;
	const char* path_Surprize;
	const char* path_Fear;
	const char* path_Angry;

    int numHappy;
    int numSad;
	int numSurprize;
	int numFear;
	int numAngry;

	argc=11;
	argv[1]= "250";
	argv[2]= "250";
	argv[3]= "250";
	argv[4]= "250";
	argv[5]= "250";
	argv[6]= "C:/Users/akash garg/Documents/Visual Studio 2012/Projects/ExpressionSVM/ExpressionSVM/Happy";
	argv[7]= "C:/Users/akash garg/Documents/Visual Studio 2012/Projects/ExpressionSVM/ExpressionSVM/Sad";
	argv[8]= "C:/Users/akash garg/Documents/Visual Studio 2012/Projects/ExpressionSVM/ExpressionSVM/Surprize";
	argv[9]= "C:/Users/akash garg/Documents/Visual Studio 2012/Projects/ExpressionSVM/ExpressionSVM/Fear";
	argv[10]= "C:/Users/akash garg/Documents/Visual Studio 2012/Projects/ExpressionSVM/ExpressionSVM/Angry";

    if(argc >= 11 )
    {
        numHappy= atoi(argv[1]);
        numSad= atoi(argv[2]);
		numSurprize= atoi(argv[3]);
	    numFear= atoi(argv[4]);
		numAngry= atoi(argv[5]);
        path_Happy= argv[6];
        path_Sad= argv[7];
		path_Surprize= argv[8];
		path_Fear= argv[9];
		path_Angry= argv[10];

    }else{
        cout << "Usage:\n" << argv[0] << " <num Happy Files> <num Sad Files> <path to Happy folder files> <path to Sad files> \n";
        return 0;
    }        

    Mat classes;//(numPlates+numNoPlates, 1, CV_32FC1);
    Mat trainingData;//(numPlates+numNoPlates, imageWidth*imageHeight, CV_32FC1 );
    Mat trainingImages;
    vector<int> trainingLabels;

    for(int i=0; i< numHappy; i++)
    {

        stringstream ss(stringstream::in | stringstream::out);
        ss << path_Happy <<"/" <<i+1 << ".png";
		cout<<ss.str()<<endl;
        //Mat img=imread(ss.str(), 0);
		Mat m = imread(ss.str(), 1);
		cout<<"Training image size"<<m.size()<<"\n";
		Mat img;
		cvtColor(m,img,CV_BGR2GRAY);
		Size size(100,100);              //Initial Size of image was 640X490
		resize(img, img, size);
        img= img.reshape(1, 1);
		cout<<"Training image size after reshape"<<img.size()<<"\n";
        trainingImages.push_back(img);
        trainingLabels.push_back(0);
    }

    for(int i=0; i< numSad; i++)
    {
        stringstream ss(stringstream::in | stringstream::out);
        ss << path_Sad <<"/" << i+1 << ".png";
        Mat img=imread(ss.str(), 0);
		Size size(100,100);              //Initial Size of image was 640X490
		resize(img, img, size);
        img= img.reshape(1, 1);
        trainingImages.push_back(img);
        trainingLabels.push_back(1);

    }

	for(int i=0; i< numSurprize; i++)
    {

        stringstream ss(stringstream::in | stringstream::out);
        ss << path_Surprize <<"/" <<i+1 << ".png";
		cout<<ss.str()<<endl;
        //Mat img=imread(ss.str(), 0);
		Mat m = imread(ss.str(), 1);
		cout<<"Training image size"<<m.size()<<"\n";
		Mat img;
		cvtColor(m,img,CV_BGR2GRAY);

		Size size(100,100);              //Initial Size of image was 640X490
		resize(img, img, size);
        img= img.reshape(1, 1);
		cout<<"Training image size after reshape"<<img.size()<<"\n";
        trainingImages.push_back(img);
        trainingLabels.push_back(2);
    }

	for(int i=0; i< numFear; i++)
    {

        stringstream ss(stringstream::in | stringstream::out);
        ss << path_Fear <<"/" <<i+1 << ".png";
		cout<<ss.str()<<endl;
        //Mat img=imread(ss.str(), 0);
		Mat m = imread(ss.str(), 1);
		cout<<"Training image size"<<m.size()<<"\n";
		Mat img;
		cvtColor(m,img,CV_BGR2GRAY);

		Size size(100,100);              //Initial Size of image was 640X490
		resize(img, img, size);
        img= img.reshape(1, 1);
		cout<<"Training image size after reshape"<<img.size()<<"\n";
        trainingImages.push_back(img);
        trainingLabels.push_back(3);
    }

	for(int i=0; i< numAngry; i++)
    {

        stringstream ss(stringstream::in | stringstream::out);
        ss << path_Angry <<"/" <<i+1 << ".png";
		cout<<ss.str()<<endl;
        //Mat img=imread(ss.str(), 0);
		Mat m = imread(ss.str(), 1);
		cout<<"Training image size"<<m.size()<<"\n";
		Mat img;
		cvtColor(m,img,CV_BGR2GRAY);

		Size size(100,100);              //Initial Size of image was 640X490
		resize(img, img, size);
        img= img.reshape(1, 1);
		cout<<"Training image size after reshape"<<img.size()<<"\n";
        trainingImages.push_back(img);
        trainingLabels.push_back(4);
    }

    Mat(trainingImages).copyTo(trainingData);
    //trainingData = trainingData.reshape(1,trainingData.rows);
    trainingData.convertTo(trainingData, CV_32FC1);
    Mat(trainingLabels).copyTo(classes);

    FileStorage fs("SVM.xml", FileStorage::WRITE);
    fs << "TrainingData" << trainingData;
    fs << "classes" << classes;
    fs.release();
	Ptr<ml::SVM> svm= ml::SVM::create();
	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(ml::SVM::CHI2);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	svm->train(trainingData, ml::ROW_SAMPLE, classes);
	svm->save("Trained.xml");
	cout<<"\n SVM classifier is trained and saved.";
	//svm= cv::Algorithm::load<ml::SVM>("Trained.xml"); // something is wrong

	//svm->load("Trained.xml");

	//cv::Ptr<cv::ml::SVM> svm2= ml::SVM::create();
    //svm2 = cv::ml::SVM::load<cv::ml::SVM>("Trained.xml");

	
	waitKey(0);
	return 0;
}

