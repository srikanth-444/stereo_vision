#include "map.h"

Map::Map(){}

void Map::createLandmarks(const Eigen::MatrixXf& objectPoints, Frame* frame, const std::vector<int>&featureIds){
    for(int i=0; i<objectPoints.rows();i++){
        Eigen::Vector3f vec = objectPoints.row(i).transpose();
        landmarks.push_back(std::make_unique<Landmark>(landmarkCounter, vec, frame, featureIds[i]));
    }
}

void Map::removeBadLandmarks(std::vector<Landmark*> landmarks){
    for(int i=0; i<landmarks.size();i++){
        if(landmarks[i]->isBad) continue;
        if (landmarks[i]->getFoundingRatio()<0.25 && landmarks[i]->nTracked<2){
            landmarks[i]->isBad=true;
            for(auto [frame,featureId]:landmarks[i]->observations){
                    frame->landmarks[featureId]=nullptr;
            }
        }
    }
}

void Map::mergeLandmarks(std::unordered_map<Landmark*, Landmark*>& mergers){
    for(auto [lm1,lm2]:mergers){
        if (lm1==nullptr ||lm2==nullptr){
            throw std::invalid_argument("Null pointers in merger"+std::to_string(lm1->id)+std::to_string(lm2->id));
        }
        if (lm1->isBad || lm2->isBad)continue;
        auto surviourLm= (lm1->observations.size() > lm2->observations.size()) ? lm1 : lm2;
        auto duplicateLm=(lm1==surviourLm)? lm2:lm1;
        for(auto[frame,featureId]:duplicateLm->observations){
            surviourLm->addObservation(frame, featureId);
            frame->landmarks[featureId]=surviourLm;
        }
    }
}

void Map::setKeyframe(std::shared_ptr<Frame> frame){
    frame->keyFrame=true;
    Frame* framePtr = frame.get();   
    keyFrames.push_back(std::move(frame));  
    for(int i=0; i<frame->landmarks.size(); i++){
         if(frame->landmarks[i]==nullptr)continue;
         frame->landmarks[i]->addObservation(framePtr, i);
    }
}

Frame* Map::getLastKeyFrame(){
    if(!keyFrames.empty())return keyFrames.back().get();
    throw std::invalid_argument("KeyFrames list is empty");
}

Frame* Map::getAgedFrame(int age){
   if(!keyFrames.empty())return keyFrames[keyFrames.size()-age].get();
   throw std::invalid_argument("KeyFrames list is empty");
}

std::vector<Frame*> Map::getClosestKeyFrames(const Frame* frame, int N){
    std::vector<Frame*> result;
    result.reserve(N);
    int i=0;
    for(auto& [f,w]:frame->covisibility){
        result.emplace_back(f);
        if(i>=N)break;
        i++;
    }
    return result;
}
std::vector<Landmark*> Map::getLocalMap(const Frame* frame){
    std::vector<Landmark*> landmarks;
    auto frames= getClosestKeyFrames(frame);
    for(auto* f : frames){
        auto lm= f->getLandmarks();
        landmarks.insert(landmarks.end(),lm.begin(), lm.end());
    }
    return landmarks;
}