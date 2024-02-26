#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <iostream>
#include <string>
#include <tuple>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include "yolov9.h"
#include "byteTrack/BYTETracker.h"
#include "common.h"
#include <random>
#include <fstream>
#include <iostream>

using namespace std;

static std::vector<cv::Scalar> colors;

bool IsPathExist(const std::string& path) {
#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return (fileAttributes != INVALID_FILE_ATTRIBUTES);
#else
    return (access(path.c_str(), F_OK) == 0);
#endif
}
bool IsFile(const std::string& path) {
    if (!IsPathExist(path)) {
        printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
        return false;
    }

#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return ((fileAttributes != INVALID_FILE_ATTRIBUTES) && ((fileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0));
#else
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
#endif
}

void format_tracker_input(cv::Mat &frame, std::vector<Detection> &detections, std::vector<byte_track::Object> &tracker_objects)
{
    const float H = 640;
    const float W = 640;
    const float r_h = H / (float)frame.rows;
    const float r_w = W / (float)frame.cols;

    for (int i = 0; i < detections.size(); i++)
    {
        float x = detections[i].box.x;
        float y = detections[i].box.y;
        float width = detections[i].box.width;
        float height = detections[i].box.height;

        if (r_h > r_w)
        {
            x = x / r_w;
            y = (y - (H - r_w * frame.rows) / 2) / r_w;
            width = width / r_w;
            height = height / r_w;
        }
        else
        {
            x = (x - (W - r_h * frame.cols) / 2) / r_h;
            y = y / r_h;
            width = width / r_h;
            height = height / r_h;
        }

        byte_track::Rect<float> rect(x, y, width, height);

        byte_track::Object obj(rect, detections[i].class_id, detections[i].confidence);

        tracker_objects.push_back(obj);
     }
}

void draw_bboxes(cv::Mat& frame, const std::vector<byte_track::BYTETracker::STrackPtr> &output)
{
    for (int i = 0; i < output.size(); i++)
    {
        auto detection = output[i];
        auto box = detection->getRect();
        auto trackId = detection->getTrackId();

        int x = detection->getRect().tlwh[0];
        int y = detection->getRect().tlwh[1];
        int width = detection->getRect().tlwh[2];
        int height = detection->getRect().tlwh[3];

        auto color_id = colors.size() % trackId;
        cv::rectangle(frame, cv::Point(x, y), cv::Point(x + width, y + height), colors[color_id], 3);

        // Detection box text
        std::string classString = std::to_string(trackId);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(x, y - 40, textSize.width + 10, textSize.height + 20);
        cv::rectangle(frame, textBox, colors[color_id], cv::FILLED);
        cv::putText(frame, classString, cv::Point(x + 5, y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }
}

int main(int argc, char** argv)
{
    const std::string engine_file_path{ argv[1] };
    const std::string path{ argv[2] };
    std::vector<std::string> imagePathList;
    bool                     isVideo{ false };
    assert(argc == 3);

    // init model
    Yolov9 yolomodel(engine_file_path);

    // init tracker
    byte_track::BYTETracker tracker(30, 30);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(100, 255);
    for (int i = 0; i < coconame.size(); i++)
    {
        cv::Scalar color = cv::Scalar(dis(gen),
            dis(gen),
            dis(gen));
        colors.push_back(color);
    }

    //path to video
    string VideoPath = path;
    // open cap
    cv::VideoCapture cap(VideoPath);

    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    // Create a VideoWriter object to save the processed video
    cv::VideoWriter output_video("output_video.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(width, height));
    while (1)
    {
        cv::Mat frame;
        std::vector<Detection> detections;
        std::vector<byte_track::Object> tracks;

        cap >> frame;

        if (frame.empty()) break;

        yolomodel.predict(frame, detections);
        
        format_tracker_input(frame, detections, tracks);
        const auto outputs = tracker.update(tracks);

        draw_bboxes(frame, outputs);

        imshow("prediction", frame);
        output_video.write(frame);
        cv::waitKey(1);
    }

    // Release resources
    cv::destroyAllWindows();
    cap.release();
    output_video.release();


    return 0;
}