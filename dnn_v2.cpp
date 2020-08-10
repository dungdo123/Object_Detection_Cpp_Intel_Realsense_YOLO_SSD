// dnn_v2.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
/*
#include <iostream>
#include <fstream>
#include <sstream>

#include <librealsense2/rs.hpp>
#include <opencv2/dnn.hpp>
#include "cv-helpers.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace rs2;


//confidence threshold
float conf_threshold = 0.5;

//nms threshold
float nms = 0.4;

// Set input image size for Yolo
const size_t inWidth = 416;
const size_t inHeight = 416;
const float WHRatio = inWidth / (float)inHeight;

vector<string> classes;

// remove unnecessary bounding boxes
void remove_box(Mat& frame, const vector<Mat>& out);

// draw bounding boxes
void draw_box(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

// get output layers
vector<String> getOutputsNames(const Net& net);

// driver function
int main(int argc, char** argv) try
{


	//1. Start streaming from Intel Realsense camera
	pipeline pipe;
	auto config = pipe.start();
	auto profile = config.get_stream(RS2_STREAM_COLOR).as<video_stream_profile>();
	rs2::align align_to(RS2_STREAM_COLOR);

	Size cropSize;
	if (profile.width() / (float)profile.height() > WHRatio)
	{
		cropSize = Size(static_cast<int>(profile.height() * WHRatio), profile.height());
	}
	else
	{
		cropSize = Size(profile.width(), static_cast<int>(profile.width() / WHRatio));
	}

	Rect crop(Point((profile.width() - cropSize.width) / 2,
		(profile.height() - cropSize.height) / 2), cropSize);
	const auto window_name = "Display Image";
	namedWindow(window_name, WINDOW_AUTOSIZE);
	
	//2. load YOLO pretrained model
	// 1. get labels of all classes
	string classesFile = "coco.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);
	
	//Load pre-trained model weights and config architecture
	String configuration = "yolov3.cfg";
	String model = "yolov3.weights";

	// load to the network
	Net net = readNetFromDarknet(configuration, model);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

    // declare a variables for input blob images
	Mat blob;

	while (getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
	{
		//Wait for the next set of frames
		auto data = pipe.wait_for_frames();

		//Make sure the frames are aligned
		data = align_to.process(data);

		auto color_frame = data.get_color_frame();
		//auto depth_frame = data.get_depth_frame();
		
		//if color frame is not updated, continue
		static int last_frame_number = 0;
		if (color_frame.get_frame_number() == last_frame_number) continue;
		last_frame_number = color_frame.get_frame_number();

		// Convert Realsense frame to openCV matrix
		auto color_mat = frame_to_mat(color_frame);

		// convert image to blob
		blobFromImage(color_mat, blob, 1 / 255, Size(inWidth, inHeight), Scalar(0, 0, 0), true, false);
		net.setInput(blob, "data");
		
		vector<Mat> outs;
		net.forward(outs, getOutputsNames(net));
		
		//Crop color frame
		color_mat = color_mat(crop);

        remove_box(color_mat, outs);
       
        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
       
        string label = format("Inference time for a frame: %.2f ms ", t);
        putText(color_mat, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
       
        Mat detectedFrame;
        color_mat.convertTo(detectedFrame, CV_8U);

		imshow(window_name, color_mat);
		if (waitKey(1) >= 0) break;
	}
	return EXIT_SUCCESS;

}
catch (const rs2::error& e)
{
	std::cerr << " Realsense error calling" << e.get_failed_function() << "(" << e.get_failed_args() << " ):\n    " << e.what() << std::endl;
	return EXIT_FAILURE;
}
catch (const std::exception& e)
{
	std::cerr << e.what() << std::endl;
	return EXIT_FAILURE;
}

//declare the used functions
// remove the boxes with low confidence scores
void remove_box(Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. 

        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > conf_threshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes

    vector<int> indices;
    NMSBoxes(boxes, confidences, conf_threshold, nms, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];

        draw_box(classIds[idx], confidences[idx], box.x, box.y,
            box.x + box.width, box.y + box.height, frame);
    }
}

// Draw the predicted bounding box
void draw_box(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}

// Get the names of the output layers
vector<Mat> outs;
vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();

        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[double(outLayers[i]) - 1];
    }
    return names;
}
*/
// This example is derived from the ssd_mobilenet_object_detection opencv demo
// and adapted to be used with Intel RealSense Cameras

#include <iostream>
#include <fstream>
#include <sstream>

#include <librealsense2/rs.hpp>
#include <opencv2/dnn.hpp>
#include "cv-helpers.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;


const size_t inWidth = 300;
const size_t inHeight = 300;
const float WHRatio = inWidth / (float)inHeight;
const float inScaleFactor = 0.007843f;
const float meanVal = 127.5;
const char* classNames[] = { "background",
                             "aeroplane", "bicycle", "bird", "boat",
                             "bottle", "bus", "car", "cat", "chair",
                             "cow", "diningtable", "dog", "horse",
                             "motorbike", "person", "pottedplant",
                             "sheep", "sofa", "train", "tvmonitor" };

int main(int argc, char** argv) try
{
    using namespace cv;
    using namespace cv::dnn;
    using namespace rs2;

    Net net = readNetFromCaffe("MobileNetSSD_deploy.prototxt",
        "MobileNetSSD_deploy.caffemodel");

    // Start streaming from Intel RealSense Camera
    pipeline pipe;
    auto config = pipe.start();
    auto profile = config.get_stream(RS2_STREAM_COLOR)
        .as<video_stream_profile>();
    rs2::align align_to(RS2_STREAM_COLOR);

    Size cropSize;
    if (profile.width() / (float)profile.height() > WHRatio)
    {
        cropSize = Size(static_cast<int>(profile.height() * WHRatio),
            profile.height());
    }
    else
    {
        cropSize = Size(profile.width(),
            static_cast<int>(profile.width() / WHRatio));
    }

    Rect crop(Point((profile.width() - cropSize.width) / 2,
        (profile.height() - cropSize.height) / 2),
        cropSize);

    const auto window_name = "Display Image";
    namedWindow(window_name, WINDOW_AUTOSIZE);

    while (getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
    {
        // Wait for the next set of frames
        auto data = pipe.wait_for_frames();
        // Make sure the frames are spatially aligned
        data = align_to.process(data);

        auto color_frame = data.get_color_frame();
        auto depth_frame = data.get_depth_frame();

        // If we only received new depth frame, 
        // but the color did not update, continue
        static int last_frame_number = 0;
        if (color_frame.get_frame_number() == last_frame_number) continue;
        last_frame_number = color_frame.get_frame_number();

        // Convert RealSense frame to OpenCV matrix:
        auto color_mat = frame_to_mat(color_frame);
        auto depth_mat = depth_frame_to_meters(depth_frame);

        Mat inputBlob = blobFromImage(color_mat, inScaleFactor,
            Size(inWidth, inHeight), meanVal, false); //Convert Mat to batch of images
        net.setInput(inputBlob, "data"); //set the network input
        Mat detection = net.forward("detection_out"); //compute output

        Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        // Crop both color and depth frames
        color_mat = color_mat(crop);
        depth_mat = depth_mat(crop);

        float confidenceThreshold = 0.8f;
        for (int i = 0; i < detectionMat.rows; i++)
        {
            float confidence = detectionMat.at<float>(i, 2);

            if (confidence > confidenceThreshold)
            {
                size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

                int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * color_mat.cols);
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * color_mat.rows);
                int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * color_mat.cols);
                int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * color_mat.rows);

                Rect object((int)xLeftBottom, (int)yLeftBottom,
                    (int)(xRightTop - xLeftBottom),
                    (int)(yRightTop - yLeftBottom));

                object = object & Rect(0, 0, depth_mat.cols, depth_mat.rows);

                // Calculate mean depth inside the detection region
                // This is a very naive way to estimate objects depth
                // but it is intended to demonstrate how one might 
                // use depth data in general
                Scalar m = mean(depth_mat(object));

                std::ostringstream ss;
                ss << classNames[objectClass] << " ";
                ss << std::setprecision(2) << m[0] << " meters away";
                String conf(ss.str());

                rectangle(color_mat, object, Scalar(0, 255, 0));
                int baseLine = 0;
                Size labelSize = getTextSize(ss.str(), FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                auto center = (object.br() + object.tl()) * 0.5;
                center.x = center.x - labelSize.width / 2;

                rectangle(color_mat, Rect(Point(center.x, center.y - labelSize.height),
                    Size(labelSize.width, labelSize.height + baseLine)),
                    Scalar(255, 255, 255), FILLED);
                putText(color_mat, ss.str(), center,
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));

                vector<double> layersTimes;
                double freq = getTickFrequency() / 1000;
                double t = net.getPerfProfile(layersTimes) / freq;
                string label = format("Inference time for a frame : %.2f ms", t);
                putText(color_mat, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
            }
        }

        imshow(window_name, color_mat);
        if (waitKey(1) >= 0) break;
    }

    return EXIT_SUCCESS;
}
catch (const rs2::error& e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
