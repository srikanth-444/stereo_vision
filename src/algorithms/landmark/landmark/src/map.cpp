#include "map.h"

Map::Map(){}

void Map::createLandmarks(const Eigen::MatrixXf& objectPoints, Frame* frame, const std::vector<int>&featureIds){
    for(int i=0; i<objectPoints.rows();i++){
        Eigen::Vector3f vec = objectPoints.row(i).transpose();
        Landmark* l = new Landmark(landmarkCounter,vec,frame,featureIds[i]);
    }
}
void removeBadLandmarks(std::vector<Landmark*> landmakrs);
void mergeLandmarks();

void setKeyframe(Frame* frame);
Frame* getLastKeyFrame();
Frame* getAgedFrames();

std::vector<Frame*>findClosestKeyFrame(const Frame* frame, int N=5);
std::vector<Landmark*>getLocalMap(const Frame* frame);