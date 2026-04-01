#include "map.h"
#include <iostream>

Map::Map(){}

std::vector<std::shared_ptr<Landmark>> Map::createLandmarks(Eigen::MatrixXf& objectPoints, std::shared_ptr<Frame> frame, const std::vector<int>&featureIds){
    
    std::vector<std::shared_ptr<Landmark>> landmarks;
    auto R=frame->q.toRotationMatrix();
    for(int i = 0; i < objectPoints.rows(); i++){
    Eigen::Vector3f vec =R* objectPoints.row(i).transpose() + frame->t;
    auto lm = std::make_shared<Landmark>(landmarkCounter, vec, frame, featureIds[i]);
    landmarkCounter++;
    landmarks.push_back(lm);
    this->landmarks.push_back(lm);
    frame->landmarks[featureIds[i]] = lm;
    }
    return landmarks;
}

void Map::removeBadLandmarks(std::vector<std::shared_ptr<Landmark>> landmarks){
    for(int i=0; i<landmarks.size();i++){
        if(landmarks[i]->isBad) continue;
        // std::cout<<"ladmark id: "<<landmarks[i]->id<<", Ratio"<<landmarks[i]->getFoundingRatio()<<std::endl;
        if (landmarks[i]->getFoundingRatio()<0.25 || landmarks[i]->nTracked<2){
            landmarks[i]->setBadFlag();
            for(auto [frame,featureId]:landmarks[i]->observations){
                    frame.lock()->landmarks[featureId].reset();
            }
            landmarks[i]->erase();
        }
    }
}


void Map::mergeLandmarks(std::unordered_map<std::shared_ptr<Landmark>, std::vector<std::shared_ptr<Landmark>>>& mergers)
{
    for (auto& [lm1, duplicates] : mergers)
    {
        if (!lm1 || lm1->isBad) continue;

        for (auto& lm2 : duplicates)
        {
            if (!lm2 || lm2->isBad) continue;

            // Choose survivor
            auto survivorLm = (lm1->observations.size() > lm2->observations.size()) ? lm1 : lm2;
            auto duplicateLm = (lm1 == survivorLm) ? lm2 : lm1;

            const auto& observations = duplicateLm->observations;

            for (const auto& [frame, featureId] : observations)
            {
                auto f = frame.lock();
                if (!f) continue;

                // Avoid duplicate observations
                if (!survivorLm->hasObservation(frame, featureId))
                {
                    survivorLm->addObservation(frame, featureId);
                }

                f->landmarks[featureId] = survivorLm;
            }

            duplicateLm->observations.clear();
            duplicateLm->isBad = true;
        }
    }
}

void Map::setKeyframe(std::shared_ptr<Frame> frame){
    frame->keyFrame=true;  
    for(int i=0; i<frame->landmarks.size(); i++){
        auto lm=frame->landmarks[i].lock();
         if(!lm)continue;
         lm->addObservation(frame, i);
    }
    keyFrames.push_back(frame);
}

std::shared_ptr<Frame> Map::getLastKeyFrame(){
    if(!keyFrames.empty())return keyFrames.back();
    throw std::invalid_argument("KeyFrames list is empty");
}

std::shared_ptr<Frame> Map::getAgedFrame(int age){
       if (age <= 0 || age >= keyFrames.size()) {
        return nullptr;
    }
    return keyFrames[keyFrames.size() - age];
}

std::vector<std::shared_ptr<Frame>> Map::getClosestKeyFrames(const std::shared_ptr<Frame> frame, int N){
    std::vector<std::shared_ptr<Frame>> result;
    result.reserve(N);
    int i=0;
    for(auto& [f,w]:frame->covisibility){
        result.emplace_back(f.lock());
        if(i>=N)break;
        i++;
    }
    return result;
}
std::vector<std::shared_ptr<Landmark>> Map::getLocalMap(const std::shared_ptr<Frame> frame) {
    std::unordered_set<int> uniqueIds;
    std::vector<std::shared_ptr<Landmark>> landmarks;

    // Current frame landmarks
    for (auto& lm : frame->getLandmarks()) {
        if (uniqueIds.insert(lm->id).second) { // true if inserted (not duplicate)
            landmarks.push_back(lm);
        }
    }

    // Keyframe landmarks
    auto frames = getClosestKeyFrames(frame);
    for (auto& f : frames) {
        for (auto& lm : f->getLandmarks()) {
            if (uniqueIds.insert(lm->id).second) {
                    landmarks.push_back(lm);
                
            }
        }
    }

    return landmarks;
}
int Map::getLengthKeyFrame(){
    return keyFrames.size();
}