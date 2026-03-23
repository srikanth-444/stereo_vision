#ifndef MAP_H
#define MAP_H
#include "frame.h"
#include "landmark.h"


class Map{
    public:
    int landmarkCounter=0;
    int FrameCounter=0;
    std::vector<std::shared_ptr<Landmark>> landmarks;
    std::vector<std::shared_ptr<Frame>> keyFrames;

    Map();
    std::vector<std::shared_ptr<Landmark>> createLandmarks(Eigen::MatrixXf& objectPoints, std::shared_ptr<Frame> frame, const std::vector<int>&featureIds);
    void removeBadLandmarks(std::vector<std::shared_ptr<Landmark>> landmakrs);
    void mergeLandmarks(std::unordered_map<std::shared_ptr<Landmark>, std::shared_ptr<Landmark>>& mergers);
    void setKeyframe(std::shared_ptr<Frame> frame);
    std::shared_ptr<Frame> getLastKeyFrame();
    std::shared_ptr<Frame> getAgedFrame(int age=2);
    std::vector<std::shared_ptr<Frame>>getClosestKeyFrames(const std::shared_ptr<Frame> frame, int N=5);
    std::vector<std::shared_ptr<Landmark>>getLocalMap(const std::shared_ptr<Frame> frame);
    int getLengthKeyFrame() ;
};
#endif