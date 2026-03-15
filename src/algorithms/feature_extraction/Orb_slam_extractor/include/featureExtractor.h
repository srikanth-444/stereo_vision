#ifndef FEATUREEXTRACTOR_H
#define FEATUREEXTRACTOR_H
#include <opencv2/core/core.hpp>

class FeatureExtractor{
    public:
    
    int operator()( cv::InputArray _image, cv::InputArray _mask,
                    std::vector<cv::KeyPoint>& _keypoints,
                    cv::OutputArray _descriptors, std::vector<int> &vLappingArea);
    int  match(const cv::Mat& lmdescriptor, const cv::Mat& descriptors, const std::vector<int>& idx);
};
#endif