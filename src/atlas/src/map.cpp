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
        if (landmarks[i]->getFoundingRatio()<0.25 || landmarks[i]->nTracked<2){
            landmarks[i]->isBad=true;
            for(auto [frame,featureId]:landmarks[i]->observations){
                    frame.lock()->landmarks[featureId].reset();
            }
            landmarks[i]->erase();
        }
    }
}

void Map::mergeLandmarks(std::unordered_map<std::shared_ptr<Landmark>, std::shared_ptr<Landmark>>& mergers){
    for(auto [lm1,lm2]:mergers){
        if (lm1==nullptr ||lm2==nullptr){
            throw std::invalid_argument("Null pointers in merger"+std::to_string(lm1->id)+std::to_string(lm2->id));
        }
        if (lm1->isBad || lm2->isBad)continue;
        auto surviourLm= (lm1->observations.size() > lm2->observations.size()) ? lm1 : lm2;
        auto duplicateLm=(lm1==surviourLm)? lm2:lm1;
        // std::cout<<surviourLm->id<<std::endl;
        auto observations = duplicateLm->observations;
        for (auto [frame, featureId] : observations) {
            surviourLm->addObservation(frame, featureId);
            frame.lock()->landmarks[featureId] = surviourLm;
        }
        duplicateLm->observations.clear();
        duplicateLm->isBad = true;
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
std::vector<std::shared_ptr<Landmark>> Map::getLocalMap(const std::shared_ptr<Frame> frame){
    std::vector<std::shared_ptr<Landmark>> landmarks;
    auto frames= getClosestKeyFrames(frame);
    auto lm =frame->getLandmarks();
        for(auto l:lm){
            // std::cout<<"frame id: "<<frame->id<<" landmark id: "<<l->id<<" decriptor: "<<l->descriptor.rows<<"x"<<l->descriptor.cols<<std::endl;
        }
        landmarks.insert(landmarks.end(),lm.begin(), lm.end());
    if(frames.empty())return landmarks;
    // std::cout<<"localMap frames: "<<frames.size()<<std::endl;
    for(int i=0; i<frames.size(); i++){
        auto lm= frames[i]->getLandmarks();
        for(auto l:lm){
            // std::cout<<"frame id: "<<frame->id<<" landmark id: "<<l->id<<" decriptor: "<<l->descriptor.rows<<"x"<<l->descriptor.cols<<std::endl;
        }
        landmarks.insert(landmarks.end(),lm.begin(), lm.end());
    }
    return landmarks;
}
int Map::getLengthKeyFrame(){
    return keyFrames.size();
}