// Visualize.h
#pragma once

#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <chrono>
#include <string>

// ─── Forward-declare your Atlas / Map / Frame types ───────────────────────────
class Atlas;
class Frame;
class Landmark;
// ─────────────────────────────────────────────────────────────────────────────

class Visualize {
public:
    // ── Construction / initialisation ────────────────────────────────────────
    explicit Visualize(Atlas* atlas);
    ~Visualize();

    // ── UNIFIED entry-point (call once per pipeline iteration) ───────────────
    void update(Frame* leftFrame,
                Frame* rightFrame = nullptr);

    // ── Legacy entry-points (names kept) ─────────────────────────────────────
    void visualize_pipeline(Frame* leftFrame, Frame* rightFrame = nullptr);
    void visualize_as_point_cloud(const Eigen::Matrix4f* T = nullptr);

    // ── Full blocking map viewer ──────────────────────────────────────────────
    void show_full_map();

private:
    // ── Internal helpers ──────────────────────────────────────────────────────
    void updateFPS();
    void updateFrame(Frame* leftFrame, Frame* rightFrame);
    cv::Mat drawFrame(Frame* frame);
    void update3D(const Eigen::Matrix4f* T);
    void addPoints(const std::vector<Eigen::Vector3f>& pts,
                   const std::vector<Eigen::Vector3f>& cols);
    void drawTrajectory();
    void drawCameraFrustum(const Eigen::Matrix4f& T, float scale = 0.2f);
    void drawPointCloud();

    // ── Atlas ─────────────────────────────────────────────────────────────────
    Atlas* atlas_;

    // ── FPS ───────────────────────────────────────────────────────────────────
    std::chrono::steady_clock::time_point prevTime_;
    double fps_       = 0.0;
    double fpsAlpha_  = 0.1;

    // ── Pre-allocated point storage ───────────────────────────────────────────
    static constexpr int kMaxPoints = 10'000'000;
    std::vector<Eigen::Vector3f> points_;   // size kMaxPoints, filled up to ptr_
    std::vector<Eigen::Vector3f> colors_;
    int ptr_ = 0;

    // ── Trajectory ────────────────────────────────────────────────────────────
    std::vector<Eigen::Vector3f> trajPoints_;

    // ── Coordinate transform (matches your Python R_transform) ───────────────
    //   [0,1,0; 1,0,0; 0,0,-1]
    Eigen::Matrix3f R_transform_;

    // ── Pangolin objects ──────────────────────────────────────────────────────
    pangolin::OpenGlRenderState* renderState_ = nullptr;
    pangolin::View*              d3dView_     = nullptr;
    bool                         firstView_   = true;
};