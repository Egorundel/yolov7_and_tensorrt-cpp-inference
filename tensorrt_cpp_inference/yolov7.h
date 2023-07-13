//
// Created by user on 11.07.2023.
//

#ifndef YOLOV7_CPP_YOLOV7_H
#define YOLOV7_CPP_YOLOV7_H

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <string>
#include <ctime>
#include <random>

#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "cuda_runtime_api.h"
#include "NvOnnxParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"

#include "logging.h"

// #define BATCH_SIZE 1
// #define IsPadding 1
// #define NUM_CLASS 13
// #define NMS_THRESH 0.45
// #define CONF_THRESH 0.25
// #define PROB_THRESH 0.80

constexpr int INPUT_W = 640;
constexpr int INPUT_H = 640;
constexpr int INPUT_SIZE = 640;

constexpr int NUM_CLASS = 11;

void* buffers[5];

const std::vector<std::string> class_names = {
		"biker",
		"car",
		"pedestrian",
		"trafficLight",
		"trafficLight-Green",
		"trafficLight-GreenLeft",
		"trafficLight-Red",
		"trafficLight-RedLeft",
		"trafficLight-Yellow",
		"trafficLight-YellowLeft",
		"truck"
};


const std::string mode = "video"; // or mode = "image",  if you want to use this code with pictures

// midpoint coordinate width and height
struct Bbox {
	float x;
	float y;
	float w;
	float h;
	float score;
	int classes;
};

float h_input[INPUT_SIZE * INPUT_SIZE * 3];
int h_output_0[1];   //1
float h_output_1[1 * 100 * 4];   //1
float h_output_2[1 * 100];   //1
float h_output_3[1 * 100];   //1

// methods
void preprocess(const cv::Mat& img, float data[]);
std::vector<Bbox> rescale_box(const std::vector<Bbox> &out, int width, int height);
cv::Mat renderBoundingBox(cv::Mat image, const std::vector<Bbox> &bboxes);

const std::array<cv::Scalar, NUM_CLASS> class_colors {
	cv::Scalar(255, 0, 0), // red for
	cv::Scalar(0, 255, 0), // lime for
	cv::Scalar(255, 69, 0), // orange red
	cv::Scalar(128, 0, 0), // maroon
	cv::Scalar(255, 215, 0), // gold
	cv::Scalar(255, 165, 0), // orange
	cv::Scalar(0, 255, 255), // aqua
	cv::Scalar(255, 255, 0), // yellow
	cv::Scalar(138, 43, 226), // blueviolet
	cv::Scalar(255, 127, 80), // coral
	cv::Scalar(0, 0, 255), // blue
};

#endif //YOLOV7_CPP_YOLOV7_H
