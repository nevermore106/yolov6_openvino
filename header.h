#pragma once

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string> 
#include "Serial.h"

struct Resize
{
    cv::Mat resized_image;
    int dw;
    int dh;
};

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

class Deal{
  public:
    Deal(){mode_ = 0;}
    Deal(int);
    
    void Get_process(std::vector<Detection>&pos);
    int Euclidean_distance(std::vector<int>a,std::vector<int>b);
    bool Is_close(Detection &pre,Detection &cur);
    bool Is_Same_Line(Detection &pre,Detection &cur);
    std::string Wrong_Number_Filter(std::vector<std::vector<char>>mat);

    std::vector<char>map;
    std::vector<std::vector<char>>record_map;
    std::string aes_output_;
    int mode_;
};

void serialThreadFunc(Serial*S1,Serial*S2);
void OpenCamera(cv::VideoCapture& capture);
Resize resize_and_pad(cv::Mat& img, cv::Size new_shape);
