#ifndef MAP_H
#define MAP_H
#include "frame.h"
#include "landmark.h"
#include <unordered_set>


class Optimizer;

class Map{
    public:
    int landmarkCounter=0;
    int FrameCounter=0;
    std::vector<std::shared_ptr<Landmark>> landmarks;
    std::vector<std::shared_ptr<Frame>> keyFrames;
    std::unordered_map<DBoW3::EntryId, std::shared_ptr<Frame>> dbIdToFrame;
    Optimizer* optimizer;

    Map(Optimizer* optimizer);
    std::vector<std::shared_ptr<Landmark>> createLandmarks(Eigen::MatrixXf& objectPoints, std::shared_ptr<Frame> frame, const std::vector<int>&featureIds);
    void removeBadLandmarks(std::vector<std::shared_ptr<Landmark>> landmakrs);
    void mergeLandmarks(std::unordered_map<std::shared_ptr<Landmark>, std::vector<std::shared_ptr<Landmark>>>& mergers);
    void setKeyframe(std::shared_ptr<Frame> frame);
    std::shared_ptr<Frame> getLastKeyFrame();
    std::shared_ptr<Frame> getAgedFrame(int age=2);
    std::vector<std::shared_ptr<Frame>>getClosestKeyFrames(const std::shared_ptr<Frame> frame, int N=5);
    std::vector<std::shared_ptr<Landmark>>getLocalMap(const std::shared_ptr<Frame> frame);
    int getLengthKeyFrame();
    void mapping(std::shared_ptr<Frame> frame,std::vector<std::shared_ptr<Landmark>>& newLandmarks);
};
#endif