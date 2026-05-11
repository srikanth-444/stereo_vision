#pragma once
#include <array>
#include <stdexcept>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Dense>
#include "frame.h"

// ── Forward declarations ──────────────────────────────────────────────────────

class CameraInterface;
class FeatureExtractor;
class Camera {
public:
    Camera(const Eigen::Matrix3f&   intrinsic,
           const Eigen::Vector4f&   distortion,
           const Eigen::Matrix4f&   extrinsic,       // 4×4 float
           std::shared_ptr<CameraInterface>         interface,
           std::shared_ptr<FeatureExtractor>        feature_extractor,
            int w, int h);

    // Getters / setters
    Eigen::Matrix3f  getIntrinsic()  const;
    Eigen::Matrix4f  getExtrinsic()  const;
    int getWidth()  const;
    int getHeight()  const;
    std::shared_ptr<FeatureExtractor> getFeatureExtractor()const;
    // Frame ops
    std::shared_ptr<Frame> getFrame(int i);
    cv::Mat undistortFrame(const cv::Mat& frame) const;
    Eigen::Vector2f project(const Eigen::Vector3f& p_cam) const;

private:
    Eigen::Matrix3f intrinsic;
    Eigen::Matrix4f extrinsic;
    Eigen::Vector4f dist_coefficients;
    std::shared_ptr<CameraInterface>  interface;
    std::shared_ptr<FeatureExtractor> feature_extractor;
    int               W, H;
    Eigen::Matrix3f new_intrinsic;
};