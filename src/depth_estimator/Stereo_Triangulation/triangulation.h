#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp> // Critical for cv::eigen2cv
#include <frame.h>
#include "featureExtractor.h"
class Stereo {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Stereo(Eigen::Matrix3f left_intrinsic,Eigen::Matrix4f left_extrinsic,Eigen::Vector4f left_dist,
                Eigen::Matrix3f right_intrinsic,Eigen::Matrix4f right_extrinsic,Eigen::Vector4f right_dist,int W, int H);
    void getDepth(
    std::shared_ptr<Frame> &leftFrame,
    std::shared_ptr<Frame> &rightFrame,
    float epipolar_threshold,
    std::vector<Eigen::Vector3f> &points_3d,
    std::vector<int> &featureIds
    );
    void rectifyPoints(std::shared_ptr<Frame> &leftFrame,
    std::shared_ptr<Frame> &rightFrame);
    void rectifyImage(std::shared_ptr<Frame> &frame);
    void visualizeMatchesBlackCanvas(
    const std::vector<cv::Point2f>& rect_l,
    const std::vector<cv::Point2f>& rect_r,
    const std::vector<std::pair<int,int>>& matches,
    int canvas_width,
    int canvas_height);
    Eigen::Matrix3f l_K;
    Eigen::Matrix4f l_extrinsic;
    Eigen::Vector4f l_dist;
    Eigen::Matrix3f r_K;
    Eigen::Matrix4f r_extrinsic;
    Eigen::Vector4f r_dist;
    Eigen::Matrix3f R_rel;
    Eigen::Vector3f t_rel;
    cv::Mat Q, R1, R2, P1, P2;
    cv::Mat cv_l_K, cv_r_K, cv_R_rel, cv_T_rel,cv_l_dist,cv_r_dist;

    float fx ;
    float fy ;
    float cx ;
    float cy ;
    float baseline ;
};