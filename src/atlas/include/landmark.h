#ifndef LANDMARK_H
#define LANDMARK_H

#include <Eigen/Dense>
#include "utils.h"
#include <opencv2/core/core.hpp>
#include <cstdint>



class Frame;

class Landmark{
    
    public:

    int id;
    cv::Mat descriptor;
    std::unordered_map<std::weak_ptr<Frame>, int, WeakPtrHash, WeakPtrEqual> observations;
    Eigen::Vector3f point3D;
    Eigen::Vector3f normal;
    float refDepth;
    int nVisible=1;
    int nTracked=0;
    int isBad;

    Landmark(int id, const Eigen::Vector3f Point3D, std::shared_ptr<Frame> frame, int featureId);
    void setNormal(const Eigen::Vector3f& cameraCenter);
    void setReferenceDepth(const Eigen::Vector3f& cameraCenter);
    bool isVisible(const Eigen::Vector3f& cameraCenter,float distThresh=2, float angleThresh=60);
    void addObservation(std::weak_ptr<Frame> frame, int featureId);
    void erase();
    void updateDescriptor();  
    void increaseVisible();
    void increaseTracked();
    float getFoundingRatio(); 
    void setBadFlag(); 
};

#endif