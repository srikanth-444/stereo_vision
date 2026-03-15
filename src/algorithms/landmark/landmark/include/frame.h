#ifndef FRAME_H
#define FRAME_H

#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <unordered_map>



class Landmark;
class FeatureExtractor;
class Frame{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int id=-1;
    int timeStamp =0;
    bool keyFrame =false;
    
    cv::Mat image;
    FeatureExtractor* featureExtractor;
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptors;

    std::vector<Landmark*> landmarks;
    std::unordered_map<Frame*, int> covisibility;
    std::unordered_map<Landmark*, Landmark*> mergers; 

    Eigen::Quaternionf R=Eigen::Quaternionf::Identity();
    Eigen::Vector3f t= Eigen::Vector3f::Constant(std::nanf(""));
    Eigen::Vector3f cameraCenter = Eigen::Vector3f::Constant(std::nanf("")); 
    Eigen::Matrix4f Tcw=Eigen::Matrix4f::Identity(); 
    Eigen::Matrix3f intrinsic;
    Eigen::Matrix4f extrinsic;

    //constructor
    Frame::Frame(int id, const cv::Mat& image, int timeStamp, Eigen::Matrix3f intrinsic,Eigen::Matrix4f extrinsic, FeatureExtractor* featureExtractor);

    //Getters
    std::vector<Eigen::Vector2f> getNotAssociatedPoints() const;
    cv::Mat getNotAssociatedDescriptors() const;
    std::vector<Landmark*> getLandmarks() const;
    std::vector<Eigen::Vector2f>getTrackedPoints() const;
    Eigen::Vector3f getCameraCenter() const;

    //Setters
    void setCameraWorldPose(const Eigen::Quaternionf& R, const Eigen::Vector3f& t);
    void setKeyPoints(std::vector<cv::KeyPoint>& keypoints);
    void setDescriptors(cv::Mat& descriptors);
    
    void updateCovisibility();
    bool projectionMatch(std::vector<Landmark*>landmarks, std::vector<cv::Point3f>& mObjectPoints, std::vector<cv::Point2f>& mImagePoints);
    void projectionMatch(std::vector<Landmark*>landmarks);
    

    private:
    std::vector<std::vector<std::vector<int>>> grid; // grid[col][row] = indices of keypoints
    int gridCols, gridRows;
    float gridElementWidthInv, gridElementHeightInv;
    float minX, minY, maxX, maxY;

    std::vector<int> Frame::GetFeaturesInArea(float x, float y, float r);
    Eigen::MatrixXf projectPoints(Eigen::MatrixXf& objectPoints);
};
#endif