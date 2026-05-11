#ifndef FRAME_H
#define FRAME_H

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "utils.h"
#include <cstdint>
#include <DBoW3/DBoW3.h>
#include <shared_mutex>

class Landmark;
class FeatureExtractor;
struct IMUSample {
    double t;
    Eigen::Vector3d accel;
    Eigen::Vector3d gyro;
};
// class IMUPreIntegration{
//     public:
//         IMUPreIntegration(double sigma_a, double sigma_g,double sigma_ba, double sigma_bg,Eigen::Vector3d bias_accel = Eigen::Vector3d::Zero(),Eigen::Vector3d bias_gyro  = Eigen::Vector3d::Zero()) : sigma_a(sigma_a), sigma_g(sigma_g),
//                             sigma_ba(sigma_ba), sigma_bg(sigma_bg),
//                             bias_accel(bias_accel), bias_gyro(bias_gyro){}

        
//         Eigen::Matrix3d delta_R  = Eigen::Matrix3d::Identity();
//         Eigen::Vector3d delta_v  = Eigen::Vector3d::Zero();
//         Eigen::Vector3d delta_p  = Eigen::Vector3d::Zero();
//         Eigen::Matrix<double,15,15> cov      = Eigen::Matrix<double,15,15>::Zero();
//         Eigen::Matrix3d dp_dba   = Eigen::Matrix3d::Zero();
//         Eigen::Matrix3d dp_dbg   = Eigen::Matrix3d::Zero();
//         Eigen::Matrix3d dv_dba   = Eigen::Matrix3d::Zero();
//         Eigen::Matrix3d dv_dbg   = Eigen::Matrix3d::Zero();
//         Eigen::Matrix3d dR_dbg   = Eigen::Matrix3d::Zero();
//         double dt_sum   = 0.0;
        

//         void integrateSamples(const std::vector<IMUSample>& samples) {
//             for (int i = 0; i + 1 < samples.size(); i++) {
//                 double dt = (samples[i+1].t - samples[i].t) * 1e-9;  // ns → s
//                 if (dt <= 0) continue;
//                 _integrateStep(samples[i], samples[i+1], dt);
//             }
//         }

//         void correctBias(const Eigen::Vector3d& dba, const Eigen::Vector3d& dbg) {
//             Eigen::Vector3d rotvec = dR_dbg * dbg;
//             double angle = rotvec.norm();
//             Eigen::Vector3d axis = angle > 1e-10 ? rotvec.normalized() : Eigen::Vector3d::UnitX();

//             Eigen::Matrix3d dR_corr = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
//             delta_R = delta_R * dR_corr;
//             delta_v += dv_dba * dba + dv_dbg * dbg;
//             delta_p += dp_dba * dba + dp_dbg * dbg;
//         }

//         // accessors
//         Eigen::Matrix3d getDeltaR() const { return delta_R; }
//         Eigen::Vector3d getDeltaV() const { return delta_v; }
//         Eigen::Vector3d getDeltaP() const { return delta_p; }
//         Eigen::Matrix<double,15,15> getCovariance() const { return cov; }
//         double getDt() const { return dt_sum; }

//     private:
//         // noise params
//         double sigma_a, sigma_g, sigma_ba, sigma_bg;

//         // biases at start of window
//         Eigen::Vector3d bias_accel, bias_gyro;

//         // preintegrated increments
//         Eigen::Matrix3d delta_R;
//         Eigen::Vector3d delta_v, delta_p;

//         // covariance [δp, δv, δφ, δba, δbg]
//         Eigen::Matrix<double,15,15> cov;

//         // bias Jacobians
//         Eigen::Matrix3d dp_dba, dp_dbg;
//         Eigen::Matrix3d dv_dba, dv_dbg;
//         Eigen::Matrix3d dR_dbg;

//         double dt_sum;

//         static Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
//             Eigen::Matrix3d S;
//             S <<     0, -v.z(),  v.y(),
//                 v.z(),      0, -v.x(),
//                 -v.y(),  v.x(),      0;
//             return S;
//         }

        
// };

class Frame : public std::enable_shared_from_this<Frame>{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int id=-1;
    int64_t timeStamp =0;
    bool keyFrame =false;
    
    cv::Mat image;
    std::shared_ptr<FeatureExtractor> featureExtractor;
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptors;
    int nVisible;
    std::shared_mutex frameMutex;

    std::vector<std::weak_ptr<Landmark>> landmarks;
    std::vector<int> notAssociatedIndices;
    std::unordered_map<std::weak_ptr<Frame>, int, WeakPtrHash, WeakPtrEqual> covisibility;
    std::unordered_map<std::shared_ptr<Landmark>, std::vector<std::shared_ptr<Landmark>>> mergers; 
    
    std::shared_ptr<Frame> rFrame;
    std::shared_ptr<Frame> LastkeyFrame;

    Eigen::Matrix4f Twr = Eigen::Matrix4f::Identity();
    Eigen::Quaternionf q=Eigen::Quaternionf::Identity();
    Eigen::Vector3f t= Eigen::Vector3f::Constant(std::nanf(""));
    Eigen::Vector3f v= Eigen::Vector3f::Constant(std::nanf(""));

    Eigen::Matrix3d delta_R  = Eigen::Matrix3d::Identity();
    Eigen::Vector3d delta_v  = Eigen::Vector3d::Zero();
    Eigen::Vector3d delta_p  = Eigen::Vector3d::Zero();
    Eigen::Matrix<double,15,15> cov      = Eigen::Matrix<double,15,15>::Zero();
    Eigen::Matrix3d dp_dba   = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dp_dbg   = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dv_dba   = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dv_dbg   = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dR_dbg   = Eigen::Matrix3d::Zero();

    double sigma_a, sigma_g, sigma_ba, sigma_bg;

        // biases at start of window
    Eigen::Vector3d bias_accel, bias_gyro;



    double dt_sum   = 0.0;



    Eigen::Vector3f cameraCenter = Eigen::Vector3f::Constant(std::nanf("")); 
    Eigen::Vector3f cameraNormal = Eigen::Vector3f::Constant(std::nanf("")); 

    Eigen::Matrix4f Tcw=Eigen::Matrix4f::Identity(); 
    Eigen::Matrix3f intrinsic;
    Eigen::Matrix4f extrinsic;
    Eigen::Vector4f dist_coefficents;
    float baseline;

    DBoW3::BowVector bowVec;
    

    //constructor
    Frame(int id, cv::Mat& image, int64_t timeStamp, Eigen::Matrix3f intrinsic,Eigen::Matrix4f extrinsic,Eigen::Vector4f dist_coefficents, std::shared_ptr<FeatureExtractor> featureExtractor);

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
    void setStereoFrame(std::shared_ptr<Frame> frame);
    void addKeyPointsToGrid();
    void correctBias(const Eigen::Vector3d& dba, const Eigen::Vector3d& dbg);
    void integrateSamples(const std::vector<IMUSample>& samples);

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

        void _integrateStep(const IMUSample& s0, const IMUSample& s1, double dt) {
            // midpoint bias-corrected measurements
            Eigen::Vector3d a_mid = 0.5 * ((s0.accel - bias_accel) + (s1.accel - bias_accel));
            Eigen::Vector3d g_mid = 0.5 * ((s0.gyro  - bias_gyro)  + (s1.gyro  - bias_gyro));

            // rotation increment
            Eigen::AngleAxisd aa(g_mid.norm() * dt,
                                g_mid.norm() > 1e-10 ? g_mid.normalized()
                                                    : Eigen::Vector3d::UnitX());
            Eigen::Matrix3d dR_step = aa.toRotationMatrix();

            // update increments
            delta_p += delta_v * dt + 0.5 * (delta_R * a_mid) * dt * dt;
            delta_v += delta_R * a_mid * dt;
            delta_R  = delta_R * dR_step;

            // covariance propagation
            _propagateCov(a_mid, dR_step, dt);

            // Jacobian update
            _updateJacobians(a_mid, dR_step, dt);

            dt_sum += dt;
        }

        void _propagateCov(const Eigen::Vector3d& a_mid,
                        const Eigen::Matrix3d& dR_step, double dt) {
            Eigen::Matrix3d Ra     = delta_R;
            Eigen::Matrix3d a_skew = skew(Ra * a_mid);

            Eigen::Matrix<double,15,15> F = Eigen::Matrix<double,15,15>::Identity();
            F.block<3,3>(0, 3)  =  Eigen::Matrix3d::Identity() * dt;
            F.block<3,3>(0, 6)  = -0.5 * a_skew * dt * dt;
            F.block<3,3>(0, 9)  = -0.5 * Ra * dt * dt;
            F.block<3,3>(3, 6)  = -a_skew * dt;
            F.block<3,3>(3, 9)  = -Ra * dt;
            F.block<3,3>(6, 6)  =  dR_step.transpose();
            F.block<3,3>(6, 12) = -Eigen::Matrix3d::Identity() * dt;

            Eigen::Matrix<double,15,12> G = Eigen::Matrix<double,15,12>::Zero();
            G.block<3,3>(3, 0)  =  Ra * dt;
            G.block<3,3>(6, 3)  =  Eigen::Matrix3d::Identity() * dt;
            G.block<3,3>(9, 6)  =  Eigen::Matrix3d::Identity() * dt;
            G.block<3,3>(12, 9) =  Eigen::Matrix3d::Identity() * dt;

            Eigen::Matrix<double,12,12> Q = Eigen::Matrix<double,12,12>::Zero();
            double sa2  = sigma_a  * sigma_a  / dt;
            double sg2  = sigma_g  * sigma_g  / dt;
            double sba2 = sigma_ba * sigma_ba * dt;
            double sbg2 = sigma_bg * sigma_bg * dt;
            Q.block<3,3>(0,0) = Eigen::Matrix3d::Identity() * sa2;
            Q.block<3,3>(3,3) = Eigen::Matrix3d::Identity() * sg2;
            Q.block<3,3>(6,6) = Eigen::Matrix3d::Identity() * sba2;
            Q.block<3,3>(9,9) = Eigen::Matrix3d::Identity() * sbg2;

            cov = F * cov * F.transpose() + G * Q * G.transpose();
        }

        void _updateJacobians(const Eigen::Vector3d& a_mid,
                            const Eigen::Matrix3d& dR_step, double dt) {
            Eigen::Matrix3d Ra     = delta_R;
            Eigen::Matrix3d a_skew = skew(Ra * a_mid);

            dp_dba += dv_dba * dt - 0.5 * Ra * dt * dt;
            dp_dbg += dv_dbg * dt - 0.5 * a_skew * dR_dbg * dt * dt;
            dv_dba += -Ra * dt;
            dv_dbg += -a_skew * dR_dbg * dt;
            dR_dbg  = dR_step.transpose() * dR_dbg - Eigen::Matrix3d::Identity() * dt;
        }

        static Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
            Eigen::Matrix3d S;
            S <<     0, -v.z(),  v.y(),
                v.z(),      0, -v.x(),
                -v.y(),  v.x(),      0;
            return S;
        }

};
#endif