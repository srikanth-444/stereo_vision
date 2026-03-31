#ifndef FRAME_H
#define FRAME_H

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "utils.h"
#include <cstdint>


class Landmark;
class FeatureExtractor;
class Frame : public std::enable_shared_from_this<Frame>{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int id=-1;
    int64_t timeStamp =0;
    bool keyFrame =false;
    
    cv::Mat image;
    FeatureExtractor* featureExtractor;
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptors;
    int nVisible;


    std::vector<std::weak_ptr<Landmark>> landmarks;
    std::vector<int> notAssociatedIndices;
    std::unordered_map<std::weak_ptr<Frame>, int, WeakPtrHash, WeakPtrEqual> covisibility;
    std::unordered_map<std::shared_ptr<Landmark>, std::vector<std::shared_ptr<Landmark>>> mergers; 

    Eigen::Quaternionf q=Eigen::Quaternionf::Identity();
    Eigen::Vector3f t= Eigen::Vector3f::Constant(std::nanf(""));
    Eigen::Vector3f cameraCenter = Eigen::Vector3f::Constant(std::nanf("")); 
    Eigen::Vector3f cameraNormal = Eigen::Vector3f::Constant(std::nanf("")); 
    Eigen::Matrix4f Tcw=Eigen::Matrix4f::Identity(); 
    Eigen::Matrix3f intrinsic;
    Eigen::Matrix4f extrinsic;
    Eigen::MatrixXf projectedPoints;
    Eigen::Vector4f dist_coefficents;

    //constructor
    Frame(int id, cv::Mat& image, int64_t timeStamp, Eigen::Matrix3f intrinsic,Eigen::Matrix4f extrinsic,Eigen::Vector4f dist_coefficents, FeatureExtractor* featureExtractor);

    //Getters
    std::vector<int>getNotAssociatedIndices()const;
    Eigen::MatrixXf  getNotAssociatedPoints() const;
    std::vector<cv::KeyPoint> getNotAssociatedKeyPoints()const;
    cv::Mat getNotAssociatedDescriptors() const;
    std::vector<std::shared_ptr<Landmark>> getLandmarks() const;
    std::vector<Eigen::Vector2f>getTrackedPoints() const;
    std::vector<int>getTrackedIds()const;
    Eigen::Vector3f getCameraCenter() const;

    //Setters
    void setCameraWorldPose(Eigen::Quaternionf q,Eigen::Vector3f t);
    void addKeyPointsToGrid();
 
    
    void updateCovisibility();
    bool projectionMatch(std::vector<std::shared_ptr<Landmark>>landmarks, std::vector<cv::Point3f>& mObjectPoints, std::vector<cv::Point2f>& mImagePoints,int r);
    void projectionMatch(std::vector<std::shared_ptr<Landmark>>landmarks);
    void match(std::vector<std::shared_ptr<Landmark>>landmarks, std::vector<cv::Point3f>& mObjectPoints, std::vector<cv::Point2f>& mImagePoints);
    void extractFeatures();
    std::vector<std::shared_ptr<Landmark>>getVisibleLandmarks(std::vector<std::shared_ptr<Landmark>>&landmarks);

    

    private:
    std::vector<std::vector<std::vector<int>>> grid; // grid[col][row] = indices of keypoints
    int gridCols, gridRows;
    float gridElementWidthInv, gridElementHeightInv;
    float minX, minY, maxX, maxY;

    std::vector<int> GetFeaturesInArea(float x, float y, float r);
    Eigen::MatrixXf projectPoints(Eigen::MatrixXf& objectPoints);
};
#endif