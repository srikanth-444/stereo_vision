#pragma once
#include <opencv2/core.hpp>


class CameraInterface {
public:
    virtual ~CameraInterface() = default;
    virtual std::pair<cv::Mat, double> read_frame()        = 0;
    virtual bool                 is_available()  = 0;
    virtual void                 stop()               = 0;
};