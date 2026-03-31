#ifndef LANDMARK_H
#define LANDMARK_H

#include <Eigen/Dense>
#include "utils.h"
#include <opencv2/core/core.hpp>
#include <cstdint>



class Frame;

class Landmark{
    
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int id;
    cv::Mat descriptor;
    std::unordered_map<std::weak_ptr<Frame>, int, WeakPtrHash, WeakPtrEqual> observations;
    Eigen::Vector3f point3D =Eigen::Vector3f::Constant(std::nanf(""));
    Eigen::Vector3f normal = Eigen::Vector3f::Constant(std::nanf(""));
    float refDepth=std::nanf("");
    int nVisible=1;
    int nTracked=0;
    int isBad;
    Eigen::Vector2f projectedpoint;

    Landmark(int id, const Eigen::Vector3f Point3D, std::shared_ptr<Frame> frame, int featureId);
    void setNormal(const Eigen::Vector3f& cameraCenter);
    void setReferenceDepth(const Eigen::Vector3f& cameraCenter);
    bool isVisible(const Eigen::Vector3f& cameraCenter,const Eigen::Vector3f& cameraNormal,float distThresh=2, float angleThresh=60);
    void addObservation(std::weak_ptr<Frame> frame, int featureId);
    bool hasObservation(const std::weak_ptr<Frame>& frame, int featureId) const;
    bool removeObservation(std::shared_ptr<Frame>& frame);
    void erase();
    void updateDescriptor();  
    void increaseVisible();
    void increaseTracked();
    float getFoundingRatio(); 
    void setBadFlag(); 
};

#endif