#ifndef MAP_H
#define MAP_H
#include "frame.h"
#include "landmark.h"

class Map{
    public:
    int landmarkCounter=0;
    int keyFrameCounter=0;
    std::vector<Landmark*> landmarks;
    std::vector<Frame*> KeyFrames;

    Map::Map();

    void createLandmarks(const Eigen::MatrixXf& objectPoints, Frame* frame, const std::vector<int>&featureIds);
    void removeBadLandmarks(std::vector<Landmark*> landmakrs);
    void mergeLandmarks();

    void setKeyframe(Frame* frame);
    Frame* getLastKeyFrame();
    Frame* getAgedFrames();
    std::vector<Frame*>findClosestKeyFrame(const Frame* frame, int N=5);

    std::vector<Landmark*>getLocalMap(const Frame* frame);
};
#endif