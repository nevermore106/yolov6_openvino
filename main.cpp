#include <opencv2/dnn.hpp>
#include <openvino/openvino.hpp>
#include <chrono>
#include <thread>
#include <atomic>
#include <algorithm>
#include "header.h"

using namespace std;
 
const float SCORE_THRESHOLD = 0.7;
const float NMS_THRESHOLD = 0.5;
const float CONFIDENCE_THRESHOLD = 0.5;
const int cacl_per_pic= 10;
int pic_number = 0;
vector<cv::Point>points_set;
cv::Mat img;

int main() {
    /******************读模型*******************/
    ov::Core core;

    std::shared_ptr<ov::Model> model = core.read_model("/home/zihe6/OpenVINO/weight/best_ckpt.xml", "/home/zihe6/OpenVINO/weight/best_ckpt.bin");
    Deal deal;
    //此处需要自行修改xml和bin的路径

    /****************初始化相机*****************/
    cv::VideoCapture capture("/dev/video0");
    OpenCamera(capture);
    capture.set(cv::CAP_PROP_FRAME_WIDTH,1920);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT,1920);
  
    /****************初始化串口******************/
    Serial s0;
    s0.OpenSerial("/dev/ttyUSB0", E_BaudRate::_115200, E_DataSize::_8, E_Parity::None, E_StopBit::_1);

    Serial s1;
    s1.OpenSerial("/dev/ttyUSB1", E_BaudRate::_115200, E_DataSize::_8, E_Parity::None, E_StopBit::_1);

    Serial s2;
    s2.OpenSerial("/dev/ttyUSB2", E_BaudRate::_115200, E_DataSize::_8, E_Parity::None, E_StopBit::_1);

    std::thread serialThread(serialThreadFunc,&s0,&s1);

    /***************初始化模型*****************/
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
    // Specify input image format
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
    // Specify preprocess pipeline to input image without resizing
    ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({ 255., 255., 255. });
    //  Specify model's input layout
    ppp.input().model().set_layout("NCHW");
    // Specify output results format
    ppp.output().tensor().set_element_type(ov::element::f32);
    // Embed above steps in the graph
    model = ppp.build();
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");

    while(1){
    // /****************读图片******************/ 
    capture.read(img);

    Resize res = resize_and_pad(img, cv::Size(640, 640));
    //Create tensor from image
    float* input_data = (float*)res.resized_image.data;
    ov::Tensor input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), input_data);
 
 
    //Create an infer request for model inference 
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
 
    //Retrieve inference results 
    const ov::Tensor& output_tensor = infer_request.get_output_tensor();
    ov::Shape output_shape = output_tensor.get_shape();
    float* detections = output_tensor.data<float>();
 
 
    //Postprocessing including NMS  
    std::vector<cv::Rect> boxes;
    vector<int> class_ids;
    vector<float> confidences;
 
    for (int i = 0; i < output_shape[1]; i++) {
        float* detection = &detections[i * output_shape[2]];
 
        float confidence = detection[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {
            float* classes_scores = &detection[5];
            cv::Mat scores(1, output_shape[2] - 5, CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
 
            if (max_class_score > SCORE_THRESHOLD) {
 
                confidences.push_back(confidence);
 
                class_ids.push_back(class_id.x);
 
                float x = detection[0];
                float y = detection[1];
                float w = detection[2];
                float h = detection[3];
 
                float xmin = x - (w / 2);
                float ymin = y - (h / 2);
 
                boxes.push_back(cv::Rect(xmin, ymin, w, h));
            }
        }
    }
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    std::vector<Detection> output;
    for (int i = 0; i < nms_result.size(); i++)
    {
        Detection result;
        int idx = nms_result[i];
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
 
    //Print results and save Figure with detections
    for (int i = 0; i < output.size(); i++)
    {

        auto detection = output[i];
        auto box = detection.box;
        auto classId = detection.class_id;
        auto confidence = detection.confidence;
        float rx = (float)img.cols / (float)(res.resized_image.cols - res.dw);
        float ry = (float)img.rows / (float)(res.resized_image.rows - res.dh);
        box.x = rx * box.x;
        box.y = ry * box.y;
        box.width = rx * box.width;
        box.height = ry * box.height;
        cout << "Bbox" << i + 1 << ": Class: " << classId << " "
            << "Confidence: " << confidence << " Scaled coords: [ "
            << "cx: " << (float)(box.x + (box.width / 2)) / img.cols << ", "
            << "cy: " << (float)(box.y + (box.height / 2)) / img.rows << ", "
            << "w: " << (float)box.width / img.cols << ", "
            << "h: " << (float)box.height / img.rows << " ]" << endl;
        float xmax = box.x + box.width;
        float ymax = box.y + box.height;
        cv::rectangle(img, cv::Point(box.x, box.y), cv::Point(xmax, ymax), cv::Scalar(0, 255, 0), 3);
        cv::rectangle(img, cv::Point(box.x, box.y - 20), cv::Point(xmax, box.y), cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(img, std::to_string(classId), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0)); 
    }

    // /****************数字处理*****************/
	  deal.Get_process(output);

    deal.record_map.push_back(deal.map);

    pic_number++;
    if(pic_number % cacl_per_pic == 0) {
      Uart_Send_AT(deal.Wrong_Number_Filter(deal.record_map),&s2,"\n\nOK",0);
      deal.record_map.clear();
      pic_number -= cacl_per_pic;
    }

    //显示具体结果
    cv::namedWindow("ImageWindow", cv::WINDOW_NORMAL);
    cv::resizeWindow("ImageWindow", 800, 600);
    cv::imshow("ImageWindow", img);
    cv::waitKey(1);

    }

    cv::destroyAllWindows();
    capture.release();
    return 0;
}

void OpenCamera(cv::VideoCapture& capture) {
  if(!capture.isOpened()){
    cout<<"camera0 open filed"<<endl;
    cv::VideoCapture capture("/dev/video1");
    sleep(1);
    if(!capture.isOpened()){
      cout<<"camera1 open filed"<<endl;
      return;
    }
  }
  cout<<"camera open"<<endl;
}


Resize resize_and_pad(cv::Mat& img, cv::Size new_shape) {
    float width = img.cols;
    float height = img.rows;
    float r = float(new_shape.width / max(width, height));
    int new_unpadW = int(round(width * r));
    int new_unpadH = int(round(height * r));
    Resize resize;
    cv::resize(img, resize.resized_image, cv::Size(new_unpadW, new_unpadH), 0, 0, cv::INTER_AREA);
 
    resize.dw = new_shape.width - new_unpadW;
    resize.dh = new_shape.height - new_unpadH;
    cv::Scalar color = cv::Scalar(100, 100, 100);
    cv::copyMakeBorder(resize.resized_image, resize.resized_image, 0, resize.dh, 0, resize.dw, cv::BORDER_CONSTANT, color);
 
    return resize;
}

void serialThreadFunc(Serial*S1,Serial*S2) {

  while(1){
    Uart_Receive_Transmit(S1,S2);
    usleep(50*1000);
  }
}
