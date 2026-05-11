#include "camera.h"
#include "camera_interface.h"
#include <stdexcept>
#include <cmath>
#include <opencv2/core/eigen.hpp>
#include "featureExtractor.h"
#include <iostream>
#include <opencv2/highgui.hpp>


// ── Construction ──────────────────────────────────────────────────────────────

Camera::Camera( const Eigen::Matrix3f&  intrinsic,const Eigen::Vector4f&  distortion,const Eigen::Matrix4f&  extrinsic,       // 4×4 float
                std::shared_ptr<CameraInterface> interface,std::shared_ptr<FeatureExtractor> feature_extractor,int w, int h):  
                interface(interface),feature_extractor(feature_extractor),
                W(w), H(h),intrinsic(intrinsic),extrinsic(extrinsic),dist_coefficients(distortion)
{
    if (!interface)
        throw std::invalid_argument("CameraInterface pointer must not be null");
    
    cv::Mat K(3, 3, CV_32F);
    cv::eigen2cv(intrinsic, K); 
    
    const int n = static_cast<int>(dist_coefficients.size());
    auto dist = cv::Mat(1, n, CV_32F);
    for (int i = 0; i < n; ++i)
        dist.at<float>(0, i) = dist_coefficients(i);


    cv::Size img_size(W, H);
    cv::Rect valid_roi;
    
    auto K_new = cv::getOptimalNewCameraMatrix(K,dist,img_size,0.0,img_size,&valid_roi);
    new_intrinsic = Eigen::Matrix3f::Zero();
    cv::cv2eigen(K_new, new_intrinsic);  
}

Eigen::Matrix3f Camera::getIntrinsic() const  { return new_intrinsic; }
Eigen::Matrix4f Camera::getExtrinsic() const  { return extrinsic; }
int Camera::getWidth() const  { return W; }
int Camera::getHeight() const  { return H; }
std::shared_ptr<FeatureExtractor> Camera::getFeatureExtractor() const { return feature_extractor; }

std::shared_ptr<Frame> Camera::getFrame(int i) {
    std::pair<cv::Mat, double> frame_ts_pair = interface->read_frame();
    if (frame_ts_pair.first.empty() || frame_ts_pair.second < 0)
        return nullptr;
    auto image=undistortFrame(frame_ts_pair.first);
    auto ts=frame_ts_pair.second;
    std::shared_ptr<Frame> frame_ptr = std::make_shared<Frame>(i, image, static_cast<int64_t>(ts * 1e9), new_intrinsic, extrinsic, dist_coefficients, feature_extractor);
    return frame_ptr;
}

cv::Mat Camera::undistortFrame(const cv::Mat& frame) const {

    if ((dist_coefficients.array().abs() < 1e-9f).all())
        return frame.clone();

    cv::Mat K, D, K_new;
    eigen2cv(intrinsic, K);
    eigen2cv(new_intrinsic, K_new);

    const int n = static_cast<int>(dist_coefficients.size());
    D = cv::Mat(1, n, CV_32F);
    for (int i = 0; i < n; ++i)
        D.at<float>(0, i) = dist_coefficients(i);

    cv::Mat out;
    cv::undistort(frame, out, K, D, K_new);
    return out;
}

Eigen::Vector2f Camera::project(const Eigen::Vector3f& p_cam) const {
    const float x = p_cam.x() / p_cam.z();
    const float y = p_cam.y() / p_cam.z();
    const float fx = new_intrinsic(0, 0), cx = new_intrinsic(0, 2);
    const float fy = new_intrinsic(1, 1), cy = new_intrinsic(1, 2);
    return { fx*x + cx, fy*y + cy };
}