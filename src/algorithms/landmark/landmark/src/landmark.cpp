#include "frame.h"
#include "landmark.h"
#include <opencv2/core/core.hpp>


Landmark::Landmark(int id, const Eigen::Vector3f& point3D, Frame* frame, int featureId): id(id),point3D(point3D)
{
    
    addObservation(frame, featureId);
    setNormal(frame->getCameraCenter());
    setReferenceDepth(frame->getCameraCenter());
}

void Landmark::addObservation(Frame* frame, int featureId)
{
    observations[frame]=featureId;
}

void Landmark::setNormal(const Eigen::Vector3f& cameraCenter)
{
    normal = point3D-cameraCenter;
    normal.normalize();
}
void Landmark::setReferenceDepth(const Eigen::Vector3f& cameraCenter)
{
    refDepth = (point3D-cameraCenter).norm();
}

bool Landmark::isVisible(const Eigen::Vector3f& cameraCenter, float distThresh, float angleThresh)
{   
    Eigen::Vector3f currentVector=point3D-cameraCenter;
    float currentDist = currentVector.norm();
    currentVector.normalize();
    if (currentDist>refDepth*distThresh || currentDist<refDepth/distThresh ) return false;
    float cosThresh = std::cos(angleThresh * M_PI / 180.0f);
    float cosAngle= currentVector.dot(normal);
    if (cosAngle<cosThresh) return false;
    return true;
}

void Landmark::updateDescriptor()
{   
    cv::Mat all_descriptors;
    for (auto& [frame, featureId]: observations){
        all_descriptors.push_back(frame->descriptors.row(featureId));
    }

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
    descriptor=all_descriptors.row(bestIdx).clone();
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
        frame->landmarks[featureId]=nullptr;
    }
    setBadFlag();
}