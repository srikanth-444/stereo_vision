#include "frame.h"
#include "landmark.h"
#include <opencv2/core/core.hpp>
#include <iostream>


Landmark::Landmark(int id, const Eigen::Vector3f point3D, std::shared_ptr<Frame> frame, int featureId): id(id),point3D(point3D)
{   

    if (!point3D.allFinite()) {
        throw std::invalid_argument("Landmark: point3D contains invalid values (NaN or Inf)");
    }
    if (!frame) {
        throw std::invalid_argument("Landmark: frame pointer is null");
    }
    if (featureId < 0) {
        throw std::invalid_argument("Landmark: featureId must be non-negative");
    }
   
    addObservation(frame, featureId);
    
}

void Landmark::addObservation(std::weak_ptr<Frame> frame, int featureId)
{
    observations[frame]=featureId;
    increaseTracked();
    updateDescriptor();
    auto f=frame.lock();
    setNormal(f->cameraCenter);
    setReferenceDepth(f->cameraCenter);
}
bool Landmark::hasObservation(const std::weak_ptr<Frame>& frame, int featureId) const
{
    for (const auto& [f, fid] : observations)
    {
        if (!f.expired() && f.lock() == frame.lock() && fid == featureId)
            return true;
    }
    return false;
}

void Landmark::setNormal(const Eigen::Vector3f& cameraCenter)
{
    Eigen::Vector3f view_dir = point3D - cameraCenter;
    view_dir.normalize();
    if (normal.array().isNaN().any()){
        normal=view_dir;
        return;
    }
    normal += view_dir;
    normal.normalize();
}
void Landmark::setReferenceDepth(const Eigen::Vector3f& cameraCenter)
{
    refDepth = (point3D-cameraCenter).norm();
}

bool Landmark::isVisible(const Eigen::Vector3f& cameraCenter,const Eigen::Vector3f& cameraNormal, float distThresh, float angleThresh)
{   
    Eigen::Vector3f currentVector=point3D-cameraCenter;
    float currentDist = currentVector.norm();
    // std::cout<<"distance gating: "<<(currentDist>refDepth*distThresh)<<", "<<(currentDist<refDepth/distThresh)<<std::endl;
    if (currentDist>refDepth*distThresh || currentDist<refDepth/distThresh ) return false;
    float cosThresh = std::cos(angleThresh * M_PI / 180.0f);
    float cosAngle= cameraNormal.dot(normal);
    // std::cout<<"angle gatting: "<<(cosAngle<cosThresh)<<", "<<cosAngle<<", "<<cosThresh<<std::endl;
    if (cosAngle<cosThresh) return false;
    return true;
}

void Landmark::updateDescriptor()
{   
    cv::Mat all_descriptors;
    for (auto& [frame, featureId]: observations){
        // std::cout<<frame<<std::endl;
        auto spframe=frame.lock();
        // std::cout<<"frame decriptors row view: "<< spframe->descriptors.row(featureId).rows << "x" << spframe->descriptors.row(featureId).cols<< std::endl;
        // std::cout<<spframe->descriptors.row(featureId)<<std::endl;
        if(!spframe)continue;
        all_descriptors.push_back(spframe->descriptors.row(featureId));
    }
    // std::cout<<"all decriptors row view: "<< all_descriptors.row(0).rows << "x" << all_descriptors.row(0).cols<< std::endl;
    if(all_descriptors.empty()){
        return;
    }
    
    cv::Mat distMatrix;

    cv::batchDistance(
        all_descriptors,      // src1
        all_descriptors,      // src2
        distMatrix,           // output distances
        CV_32S,               // distance type
        cv::noArray(),        // index output (optional)
        cv::NORM_HAMMING      // norm type for ORB descriptors
    );
    int N = distMatrix.rows;
    int bestMedian = INT_MAX;
    int bestIdx = 0;
        for(int i = 0; i < N; i++)
    {
        std::vector<int> dists;

        for(int j = 0; j < N; j++)dists.push_back(distMatrix.at<int>(i, j));

        std::sort(dists.begin(), dists.end());

        int median = dists[(N-1)/2];

        if(median < bestMedian)
        {
            bestMedian = median;
            bestIdx = i;
        }
    }
    // std::cout<<all_descriptors.row(bestIdx)<<std::endl;
     all_descriptors.row(bestIdx).copyTo(this->descriptor);
    //  std::cout<<this->descriptor<<std::endl;
}

void Landmark::increaseVisible(){
    nVisible+=1;
}

void Landmark::increaseTracked(){
    nTracked+=1;
}

float Landmark::getFoundingRatio(){
    if(nVisible>0){
        return nTracked/nVisible;
    }
    else{
        return 0;
    }

}

void Landmark::setBadFlag(){
    isBad=true;
}
void Landmark::erase(){
    for(auto& [frame, featureId]: observations){
        if(frame.lock())continue;
        frame.lock()->landmarks[featureId].reset();
    }
    setBadFlag();
}