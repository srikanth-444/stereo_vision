#include "frame.h"
#include "landmark.h"
#include "featureExtractor.h"

Frame::Frame(int id, cv::Mat& image, int timeStamp, Eigen::Matrix3f intrinsic,Eigen::Matrix4f extrinsic,FeatureExtractor* featureExtractor): id(id),image(image),timeStamp(timeStamp), intrinsic(intrinsic), extrinsic(extrinsic), featureExtractor(featureExtractor){}

Eigen::Vector3f Frame::getCameraCenter()const{
    if(cameraCenter.array().isNaN().any()){
        throw std::runtime_error("Frame ID " + std::to_string(id) + " has uninitialized (NaN) camerCenter!");
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

std::vector<Eigen::Vector2f> Frame::getNotAssociatedPoints()const{
    std::vector<Eigen::Vector2f> result;
    result.reserve(keyPoints.size());
    for(int i=0; i<landmarks.size(); i++){
        if(landmarks[i]==nullptr){
            result.emplace_back(keyPoints[i].pt.x,keyPoints[i].pt.y);
        }
    }
    return result;
}
cv::Mat Frame::getNotAssociatedDescriptors()const{
    int count = 0;
    for (auto* lm : landmarks) {
        if (lm == nullptr) count++;
    }
    cv::Mat result(count, descriptors.cols, descriptors.type());
    int currentRow=0;
    for(int i=0; i<landmarks.size(); i++){
        if(landmarks[i]==nullptr){
            descriptors.row(i).copyTo(result.row(currentRow));
            currentRow++;
        }
    }
    return result;
}
std::vector<Landmark*> Frame::getLandmarks()const{
    std::vector<Landmark*> result;
    result.reserve(landmarks.size());
    for(int i=0; i<landmarks.size();i++){
        if(landmarks[i]!=nullptr){
            result.emplace_back(landmarks[i]);
        }
    }

    return result;
}

std::vector<Eigen::Vector2f> Frame::getTrackedPoints()const{
    std::vector<Eigen::Vector2f> result;
    result.reserve(keyPoints.size());
    for(int i=0; i<landmarks.size(); i++){
        if(landmarks[i]!=nullptr){
            result.emplace_back(keyPoints[i].pt.x,keyPoints[i].pt.y);
        }
    }
    return result;
}

void Frame::updateCovisibility(){
    std::vector<Landmark*> lm = getLandmarks();
    for(auto* lm:landmarks){
        for(auto& [frameptr,featureId]:lm->observations){
            if(frameptr==this)continue;
            covisibility[frameptr]++;
            frameptr->covisibility[this]++;
        }
    }

}

void Frame::setCameraWorldPose(const Eigen::Quaternionf& R, const Eigen::Vector3f& t){
    this->R = R.normalized();
    this->t = t;

    Eigen::Matrix3f RM = this->R.toRotationMatrix();

    // robot pose in world
    Eigen::Matrix4f Twr = Eigen::Matrix4f::Identity();
    Twr.block<3,3>(0,0) = RM;
    Twr.block<3,1>(0,3) = t;

    // camera pose in world
    Eigen::Matrix4f Twc = Twr * extrinsic;

    // camera center
    cameraCenter = Twc.block<3,1>(0,3);

    // projection transform
    Eigen::Matrix3f Rwc = Twc.block<3,3>(0,0);
    Eigen::Vector3f twc = Twc.block<3,1>(0,3);

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
bool Frame::projectionMatch(const std::vector<Landmark*>landmarks, std::vector<cv::Point3f>& mObjectPoints, std::vector<cv::Point2f>& mImagePoints){

    Eigen::MatrixXf objectPoints(4,landmarks.size());
    cv::Mat lmDescriptors(landmarks.size(), descriptors.cols, descriptors.type());
    int i=0;
    for(auto* lm:landmarks){
        objectPoints(0,i)=lm->point3D[0];
        objectPoints(1,i)=lm->point3D[1];
        objectPoints(2,i)=lm->point3D[2];
        objectPoints(3,i)=1.0f;
        lm->descriptor.copyTo(lmDescriptors.row(i));
        i++;
    }
    Eigen::MatrixXf projectedPoints=projectPoints(objectPoints);

    for(int i = 0; i < projectedPoints.rows(); i++)
    {
        float u = projectedPoints(i,0);
        float v = projectedPoints(i,1);

        auto idx=GetFeaturesInArea(u,v,10);
        int id=featureExtractor->match(lmDescriptors.row(i),descriptors,idx);
        if(id==-1) continue;
        if(this->landmarks[id]==nullptr){
            if(landmarks[i]->observations.find(this) == landmarks[i]->observations.end()){
                this->landmarks[id]=landmarks[i];
                this->landmarks[id]->addObservation(this,id);
                mObjectPoints.emplace_back(objectPoints(0,i),objectPoints(1,i),objectPoints(2,i));    
                mImagePoints.emplace_back(keyPoints[id].pt);
            }    
        }
    }
    if(mObjectPoints.size()<8)return false;
    
    return true;
}

void Frame::projectionMatch(const std::vector<Landmark*> landmarks){
    Eigen::MatrixXf objectPoints(4,landmarks.size());
    cv::Mat lmDescriptors(landmarks.size(), descriptors.cols, descriptors.type());
    int i=0;
    for(auto* lm:landmarks){
        objectPoints(0,i)=lm->point3D[0];
        objectPoints(1,i)=lm->point3D[1];
        objectPoints(2,i)=lm->point3D[2];
        objectPoints(3,i)=1.0f;
        lm->descriptor.copyTo(lmDescriptors.row(i));
        i++;
    }
    Eigen::MatrixXf projectedPoints=projectPoints(objectPoints);

    for(int i = 0; i < projectedPoints.rows(); i++)
    {
        float u = projectedPoints(i,0);
        float v = projectedPoints(i,1);

        auto idx=GetFeaturesInArea(u,v,10);
        int id=featureExtractor->match(lmDescriptors.row(i),descriptors,idx);
        if(id==-1) continue;
        if(this->landmarks[id]=nullptr){
            if(landmarks[i]->observations.find(this) == landmarks[i]->observations.end()){
                this->landmarks[id]=landmarks[i];
                this->landmarks[id]->addObservation(this,id);
            }    
        }
        else{
            if (this->landmarks[id]==landmarks[i]) continue;
            mergers[this->landmarks[id]]=landmarks[i];
        }
    }
    
}

void Frame::extractFeatures(){
    std::vector<int> lap_area = {0, image.cols};
    (*featureExtractor)(image, cv::noArray(), keyPoints, descriptors, lap_area);
    landmarks.resize(keyPoints.size(), nullptr);
    addKeyPointsToGrid();
}