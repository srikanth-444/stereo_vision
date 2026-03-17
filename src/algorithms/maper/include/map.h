#ifndef MAP_H
#define MAP_H
#include "frame.h"
#include "landmark.h"


class Map{
    public:
    int landmarkCounter=0;
    int FrameCounter=0;
    std::vector<std::unique_ptr<Landmark>> landmarks;
    std::vector<std::shared_ptr<Frame>> keyFrames;

    Map();
    void createLandmarks(const Eigen::MatrixXf& objectPoints, Frame* frame, const std::vector<int>&featureIds);
    void removeBadLandmarks(std::vector<Landmark*> landmakrs);
    void mergeLandmarks(std::unordered_map<Landmark*, Landmark*>& mergers);
    void setKeyframe(std::shared_ptr<Frame> frame);
    Frame* getLastKeyFrame();
    Frame* getAgedFrame(int age=2);
    std::vector<Frame*>getClosestKeyFrames(const Frame* frame, int N=5);
    std::vector<Landmark*>getLocalMap(const Frame* frame);
};
#endif