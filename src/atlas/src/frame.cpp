#include "frame.h"
#include "landmark.h"
#include "featureExtractor.h"
#include <numeric>
#include <iostream>

Frame::Frame(int id, cv::Mat& image, int64_t timeStamp, Eigen::Matrix3f intrinsic,Eigen::Matrix4f extrinsic,FeatureExtractor* featureExtractor): 
id(id),image(image),timeStamp(timeStamp), intrinsic(intrinsic), extrinsic(extrinsic), featureExtractor(featureExtractor){
}

Eigen::Vector3f Frame::getCameraCenter()const{
    if(cameraCenter.array().isNaN().any()){
        throw std::runtime_error("Frame ID " + std::to_string(id) + " has uninitialized (NaN) cameraCenter!");
    }
    return cameraCenter;
}
void Frame::addKeyPointsToGrid()
{
    // Grid parameters
    gridCols = 64;
    gridRows = 48;
    minX = 0; maxX = image.cols;
    minY = 0; maxY = image.rows;

    gridElementWidthInv = gridCols / (maxX - minX);
    gridElementHeightInv = gridRows / (maxY - minY);

    grid.resize(gridCols, std::vector<std::vector<int>>(gridRows));

    // Assign keypoints to grid
    for(int i = 0; i < keyPoints.size(); i++)
    {
        int c = std::min(gridCols-1, std::max(0, int((keyPoints[i].pt.x - minX) * gridElementWidthInv)));
        int r = std::min(gridRows-1, std::max(0, int((keyPoints[i].pt.y - minY) * gridElementHeightInv)));
        grid[c][r].push_back(i);
    }
}
std::vector<int> Frame::getNotAssociatedIndices()const{
    return notAssociatedIndices;
                
}
Eigen::MatrixXf Frame::getNotAssociatedPoints()const{
    Eigen::MatrixXf result(notAssociatedIndices.size(), 2);
    int row = 0;
    for (int i : notAssociatedIndices) {
        if (!landmarks[i].lock()) {
            result.row(row) << keyPoints[i].pt.x, keyPoints[i].pt.y;
            row++;
        }
    }
    return result.topRows(row); 
}
cv::Mat Frame::getNotAssociatedDescriptors()const{
    
    cv::Mat result(notAssociatedIndices.size(), descriptors.cols, descriptors.type());
    int currentRow=0;
    for(auto i:notAssociatedIndices){
        auto lm=landmarks[i].lock();
        if(!lm){
            descriptors.row(i).copyTo(result.row(currentRow));
            currentRow++;
        }
    }
    return result;
}
std::vector<std::shared_ptr<Landmark>> Frame::getLandmarks()const{
    std::vector<std::shared_ptr<Landmark>> result;
    result.reserve(landmarks.size());
    for(int i=0; i<landmarks.size();i++){
        auto lm=landmarks[i].lock();
        if(lm){
            result.emplace_back(lm);
            // std::cout<<"frame id: "<<this->id<<" landmark id " <<landmarks[i].lock()->id<<std::endl;
        }
    }

    return result;
}

std::vector<Eigen::Vector2f> Frame::getTrackedPoints()const{
    std::vector<Eigen::Vector2f> result;
    result.reserve(keyPoints.size());
    for(int i=0; i<landmarks.size(); i++){
        auto lm=landmarks[i].lock();
        if(lm){
            result.emplace_back(keyPoints[i].pt.x,keyPoints[i].pt.y);
        }
    }
    return result;
}

void Frame::updateCovisibility(){
    
    // std::cout<<landmarks.size()<<std::endl;
    for(int i=0; i<landmarks.size(); i++){
        auto lm=landmarks[i].lock();
        if(lm){
            for(auto [frameptr,featureId]:lm->observations){
                // std::cout<<frameptr<<std::endl;
                std::shared_ptr<Frame> wthis = shared_from_this();
                auto spFrame = frameptr.lock();
                if (!spFrame) continue;               // skip expired
                if (spFrame.get() == this) continue;  // skip self
                covisibility[spFrame]++;
                spFrame->covisibility[wthis]++;
            }
        }
    }

}

void Frame::setCameraWorldPose(Eigen::Quaternionf q,Eigen::Vector3f t){
    this->q=q;
    this->t=t;
    q.normalize();
    auto R=q.toRotationMatrix();
    Eigen::Matrix4f Twr = Eigen::Matrix4f::Identity();
    Twr.block<3,3>(0,0)=R;
    Twr.block<3,1>(0,3)=t;
    Eigen::Matrix4f Twc = Twr * extrinsic;
    cameraCenter = Twc.block<3,1>(0,3);

    Eigen::Matrix3f Rwc = Twc.block<3,3>(0,0);
    Eigen::Vector3f twc = Twc.block<3,1>(0,3);
    cameraNormal=Rwc*Eigen::Vector3f(0,0,1);
    cameraNormal.normalize();
    Eigen::Matrix3f Rcw = Rwc.transpose();
    Eigen::Vector3f tcw = -Rcw * twc;

    this->Tcw.setIdentity();
    this->Tcw.block<3,3>(0,0) = Rcw;
    this->Tcw.block<3,1>(0,3) = tcw;
}

std::vector<int> Frame::GetFeaturesInArea(float x, float y, float r)
{
    std::vector<int> vIndices;

    int minC = std::max(0, int((x - r - minX) * gridElementWidthInv));
    int maxC = std::min(gridCols-1, int((x + r - minX) * gridElementWidthInv));
    int minR = std::max(0, int((y - r - minY) * gridElementHeightInv));
    int maxR = std::min(gridRows-1, int((y + r - minY) * gridElementHeightInv));

    for(int c = minC; c <= maxC; c++)
    {
        for(int r = minR; r <= maxR; r++)
        {
            for(int idx : grid[c][r])
            {
                float dx = keyPoints[idx].pt.x - x;
                float dy = keyPoints[idx].pt.y - y;
                if(std::abs(dx) < r && std::abs(dy) < r)
                    vIndices.push_back(idx);
            }
        }
    }

    return vIndices;
}


Eigen::MatrixXf Frame::projectPoints(Eigen::MatrixXf& objectPoints){
    int N=objectPoints.cols();
    Eigen::MatrixXf imagePoints(N,2);
    Eigen::MatrixXf pc = Tcw.block<3,4>(0,0) * objectPoints;
    Eigen::MatrixXf uc= intrinsic*pc;
    uc.row(0).array()/=uc.row(2).array();
    uc.row(1).array()/=uc.row(2).array();
    imagePoints.col(0)=uc.row(0).transpose();
    imagePoints.col(1)=uc.row(1).transpose();
    return imagePoints;

}
bool Frame::projectionMatch(const std::vector<std::shared_ptr<Landmark>>landmarks, std::vector<cv::Point3f>& mObjectPoints, std::vector<cv::Point2f>& mImagePoints){
    // std::cout<<"entered the main function"<<std::endl;
    
    Eigen::MatrixXf objectPoints(4,landmarks.size());
    for(int i=0; i<landmarks.size();i++)
    {
        float u = landmarks[i]->projectedpoint(0);
        float v = landmarks[i]->projectedpoint(1);
        auto idx=GetFeaturesInArea(u,v,10);
        int id=featureExtractor->match(landmarks[i]->descriptor,descriptors,idx,0.9);
        if(id==-1) continue;
        if(!this->landmarks[id].lock()){
            std::weak_ptr<Frame> wthis = shared_from_this();
            if(landmarks[i]->observations.find(wthis) == landmarks[i]->observations.end()){
                this->landmarks[id]=landmarks[i];
                // this->landmarks[id]->addObservation(this,id);
                auto it = std::find(notAssociatedIndices.begin(), notAssociatedIndices.end(), id);
                if (it != notAssociatedIndices.end()) {
                    notAssociatedIndices.erase(it);
                }
                mObjectPoints.emplace_back(objectPoints(0,i),objectPoints(1,i),objectPoints(2,i));    
                mImagePoints.emplace_back(keyPoints[id].pt);
            }    
        }
    }
    if(mObjectPoints.size()<8)return false;
    
    return true;
}
std::vector<std::shared_ptr<Landmark>>Frame::getVisibleLandmarks(std::vector<std::shared_ptr<Landmark>>& landmarks)
{
    std::vector<std::shared_ptr<Landmark>> newLandmarks;
    newLandmarks.reserve(landmarks.size());

    Eigen::Matrix3f Rcw = (Tcw.block<3,3>(0,0)).eval();
    Eigen::Vector3f tcw = (Tcw.block<3,1>(0,3)).eval();

    for (const auto& lm : landmarks)
    {
        // 1. Viewing angle check
        if (!lm->isVisible(this->cameraCenter, this->cameraNormal))
            continue;

        // 2. Transform to camera frame
        Eigen::Vector3f p = Rcw * lm->point3D + tcw;

        // 3. Depth check
        if (p[2] <= 0)
            continue;

        // 4. Projection
        Eigen::Vector3f proj = this->intrinsic * p;
        proj /= proj[2];

        float u = proj[0];
        float v = proj[1];

        // 5. Image bounds check
        if (u < 0 || u >= image.cols || v < 0 || v >= image.rows)
            continue;

        // 6. Store result
        lm->projectedpoint = proj.head<2>();
        lm->increaseVisible();
        newLandmarks.emplace_back(lm);
    }

    nVisible = newLandmarks.size();
    return newLandmarks;
}

void Frame::projectionMatch(const std::vector<std::shared_ptr<Landmark>> landmarks){
    Eigen::MatrixXf objectPoints(4,landmarks.size());

    
    for(int i=0; i<landmarks.size();i++){
    
        float u = landmarks[i]->projectedpoint(0);
        float v = landmarks[i]->projectedpoint(1);

        auto idx=GetFeaturesInArea(u,v,50);
        int id=featureExtractor->match(landmarks[i]->descriptor,descriptors,idx,0.9);
        if(id==-1) continue;
        if(!this->landmarks[id].lock()){
            std::weak_ptr<Frame> wthis = shared_from_this();
            if(landmarks[i]->observations.find(wthis) == landmarks[i]->observations.end()){
                this->landmarks[id]=landmarks[i];
                auto it = std::find(notAssociatedIndices.begin(), notAssociatedIndices.end(), id);
                if (it != notAssociatedIndices.end()) {
                    notAssociatedIndices.erase(it);
                }
            }    
        }
        else{
            if (this->landmarks[id].lock() && this->landmarks[id].lock() ==landmarks[i]) continue;
            mergers[this->landmarks[id].lock()].push_back(landmarks[i]);
            
        }
    }
    
}
void Frame::match(const std::vector<std::shared_ptr<Landmark>>landmarks, std::vector<cv::Point3f>& mObjectPoints, std::vector<cv::Point2f>& mImagePoints){

    std::vector<int> idx;
    idx.resize(descriptors.rows);
    std::iota(idx.begin(), idx.end(),0);
    for(int i=0; i<landmarks.size();i++){
        // std::cout << "lm->descriptor: "<<landmarks[i]->id<<" shape "<<landmarks[i]->descriptor.rows << "x" << landmarks[i]->descriptor.cols  << std::endl;
        int id=featureExtractor->match(landmarks[i]->descriptor,descriptors,idx,0.9);
        if(id==-1) continue;
        // std::cout<<(this->landmarks[id]==nullptr)<<std::endl;
        if(!this->landmarks[id].lock()){
            std::weak_ptr<Frame> wthis = shared_from_this();
            if(landmarks[i]->observations.find(wthis) == landmarks[i]->observations.end()){
                this->landmarks[id]=landmarks[i];
                auto it = std::find(notAssociatedIndices.begin(), notAssociatedIndices.end(), id);
                if (it != notAssociatedIndices.end()) {
                    notAssociatedIndices.erase(it);
                }
                mObjectPoints.emplace_back(landmarks[i]->point3D[0],landmarks[i]->point3D[1],landmarks[i]->point3D[2]);    
                mImagePoints.emplace_back(keyPoints[id].pt);
            }    
        }  
    }
}

void Frame::extractFeatures(){
    std::vector<int> lap_area = {0, image.cols};
    (*featureExtractor)(image, cv::noArray(), keyPoints, descriptors, lap_area);
    landmarks.resize(keyPoints.size());
    notAssociatedIndices.resize(keyPoints.size());
    std::iota(notAssociatedIndices.begin(), notAssociatedIndices.end(), 0); // fill 0..N-1
    addKeyPointsToGrid();
}