// Visualize.cpp
#include "visualize.h"
#include "atlas.h"
#include "frame.h"
#include "landmark.h"
#include <iostream>

// ─────────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────────
Visualize::Visualize(Atlas* atlas)
    : atlas_(atlas),
      prevTime_(std::chrono::steady_clock::now())
{
    R_transform_ << 0.f,  1.f,  0.f,
                    -1.f,  0.f,  0.f,
                    0.f,  0.f, 1.f;

    points_.resize(kMaxPoints, Eigen::Vector3f::Zero());
    colors_.resize(kMaxPoints, Eigen::Vector3f::Zero());

    pangolin::CreateWindowAndBind("Map + Trajectory", 752, 480);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    renderState_ = new pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(752, 480, 500, 500, 376, 240, 0.1f, 10000.f),
        pangolin::ModelViewLookAt(0, -5, -10,
                                  0,  0,   0,
                                  0, -1,   0)
    );

    d3dView_ = &pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, -752.0f / 480.0f)
        .SetHandler(new pangolin::Handler3D(*renderState_));
}

Visualize::~Visualize() {
    delete renderState_;
    cv::destroyAllWindows();
}

// ─────────────────────────────────────────────────────────────────────────────
// UNIFIED entry-point
// ─────────────────────────────────────────────────────────────────────────────
void Visualize::update(Frame* leftFrame,
                       Frame* rightFrame)
{
    updateFPS();
    updateFrame(leftFrame, rightFrame);
    update3D(&leftFrame->Twr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Legacy wrappers
// ─────────────────────────────────────────────────────────────────────────────
void Visualize::visualize_pipeline(Frame* leftFrame, Frame* rightFrame) {
    updateFPS();
    updateFrame(leftFrame, rightFrame);
}

void Visualize::visualize_as_point_cloud(const Eigen::Matrix4f* T) {
    update3D(T);
}

// ─────────────────────────────────────────────────────────────────────────────
// FPS
// ─────────────────────────────────────────────────────────────────────────────
void Visualize::updateFPS() {
    auto now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(now - prevTime_).count();
    prevTime_ = now;
    if (dt > 0.f)
        fps_ = (1.f - fpsAlpha_) * fps_ + fpsAlpha_ * (1.f / dt);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2-D frame window (OpenCV)
// ─────────────────────────────────────────────────────────────────────────────
void Visualize::updateFrame(Frame* leftFrame, Frame* rightFrame) {
    cv::Mat display = drawFrame(leftFrame);

    if (rightFrame)
        cv::hconcat(display, drawFrame(rightFrame), display);

    std::string fpsStr = "FPS: " + std::to_string(fps_).substr(0, 5);
    cv::putText(display, fpsStr, {10, 30},
                cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 0, 255}, 2, cv::LINE_AA);

    cv::Mat small;
    cv::resize(display, small, {}, 0.9, 0.9);
    cv::imshow("Frame", small);
    cv::waitKey(1);
}

cv::Mat Visualize::drawFrame(Frame* frame) {
    cv::Mat img = frame->image.clone();
    if (img.channels() == 1)
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);

    auto trackedPts = frame->getTrackedPoints();
    auto landmarks  = frame->getLandmarks();

    for (size_t i = 0; i < trackedPts.size(); ++i) {
        cv::Point pt  = cv::Point((int)trackedPts[i][0],                    (int)trackedPts[i][1]);
        cv::Point pt1 = cv::Point((int)landmarks[i]->projectedpoint[0],     (int)landmarks[i]->projectedpoint[1]);

        cv::circle(img, pt,  2, {0, 255, 0}, -1);
        cv::circle(img, pt1, 2, {255, 0, 0},  1);
        cv::line  (img, pt, pt1, {0, 0, 255},  2);
    }
    return img;
}

// ─────────────────────────────────────────────────────────────────────────────
// 3-D update (Pangolin)
// ─────────────────────────────────────────────────────────────────────────────
void Visualize::update3D(const Eigen::Matrix4f* T) {
    ptr_ = 0;  // reset

    auto activeMap = atlas_->getActiveMap();

    std::vector<Eigen::Vector3f> pts;
    std::vector<Eigen::Vector3f> cols;
    for (auto lm : activeMap->landmarks) {
        if (!lm || lm->isBad) continue;
        pts.push_back(lm->point3D);
        cols.push_back(Eigen::Vector3f::Zero());
    }
    addPoints(pts, cols);

    {
        std::vector<Eigen::Vector3f> pts;
        for (auto lm : activeMap->landmarks)
            if (lm && !lm->isBad) pts.push_back(lm->point3D);
        std::vector<Eigen::Vector3f> cols(pts.size(), Eigen::Vector3f::Zero());
        addPoints(pts, cols);
    }

    if (T) {
        Eigen::Vector3f t = T->block<3,1>(0,3);
        Eigen::Vector3f tViz = R_transform_ * t;
        trajPoints_.push_back(tViz);
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    d3dView_->Activate(*renderState_);

    drawPointCloud();
    drawTrajectory();

    if (T) drawCameraFrustum(*T);

    pangolin::FinishFrame();
}

// ─────────────────────────────────────────────────────────────────────────────
// Add points to pre-allocated buffer
// ─────────────────────────────────────────────────────────────────────────────
void Visualize::addPoints(const std::vector<Eigen::Vector3f>& pts,
                          const std::vector<Eigen::Vector3f>& cols)
{
    int n = static_cast<int>(pts.size());
    if (ptr_ + n >= kMaxPoints) {
        std::cerr << "[Visualize] Max points exceeded!\n";
        return;
    }
    for (int i = 0; i < n; ++i) {
        points_[ptr_ + i] = R_transform_ * pts[i];
        colors_[ptr_ + i] = cols[i];
    }
    ptr_ += n;
}

// ─────────────────────────────────────────────────────────────────────────────
// Draw global point cloud
// ─────────────────────────────────────────────────────────────────────────────
void Visualize::drawPointCloud() {
    if (ptr_ == 0) return;

    glPointSize(2.0f);
    glBegin(GL_POINTS);
    for (int i = 0; i < ptr_; ++i) {
        glColor3fv(colors_[i].data());
        glVertex3fv(points_[i].data());
    }
    glEnd();

    auto lastKF = atlas_->getActiveMap()->getLastKeyFrame();
    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_POINTS);
    for (auto lm : lastKF->getLandmarks()) {
        if (!lm) continue;
        Eigen::Vector3f p = R_transform_ * lm->point3D;
        glVertex3fv(p.data());
    }
    glEnd();
}

// ─────────────────────────────────────────────────────────────────────────────
// Draw trajectory polyline
// ─────────────────────────────────────────────────────────────────────────────
void Visualize::drawTrajectory() {
    if (trajPoints_.size() < 2) return;

    glLineWidth(2.0f);
    glColor3f(0.0f, 0.0f, 1.0f);
    glBegin(GL_LINE_STRIP);
    for (auto& p : trajPoints_)
        glVertex3fv(p.data());
    glEnd();
}

// ─────────────────────────────────────────────────────────────────────────────
// Camera frustum
// ─────────────────────────────────────────────────────────────────────────────
void Visualize::drawCameraFrustum(const Eigen::Matrix4f& T, float scale) {
    Eigen::Matrix3f R_cam = T.block<3,3>(0,0);
    Eigen::Vector3f t_cam = T.block<3,1>(0,3);

    Eigen::Matrix3f Rv = R_transform_ * R_cam;
    Eigen::Vector3f tv = R_transform_ * t_cam;

    Eigen::Matrix<float,5,3> corners;
    corners << 0.0f,  0.0f, 0.0f,
               0.2f,  0.3f, 0.5f,
               0.2f, -0.3f, 0.5f,
              -0.2f, -0.3f, 0.5f,
              -0.2f,  0.3f, 0.5f;
    corners *= scale;

    Eigen::Matrix<float,5,3> w;
    for (int i = 0; i < 5; ++i)
        w.row(i) = (Rv * corners.row(i).transpose()).transpose() + tv.transpose();

    int lines[8][2] = {{0,1},{0,2},{0,3},{0,4},{1,2},{2,3},{3,4},{4,1}};

    glLineWidth(2.0f);
    glColor3f(0.0f, 0.5f, 1.0f);
    glBegin(GL_LINES);
    for (auto& l : lines) {
        glVertex3fv(w.row(l[0]).data());
        glVertex3fv(w.row(l[1]).data());
    }
    glEnd();
}

// ─────────────────────────────────────────────────────────────────────────────
// Full blocking map viewer
// ─────────────────────────────────────────────────────────────────────────────
void Visualize::show_full_map() {
    auto maps = atlas_->maps;
    std::vector<Eigen::Vector3f> pts;
    std::vector<Eigen::Vector3f> kfPts;
    for(auto activeMap:maps){
        
        for (auto lm : activeMap->landmarks) {
            if (!lm || lm->isBad) continue;
            pts.push_back(R_transform_ * lm->point3D);
        }
        if (pts.empty()) { std::cout << "No landmarks to display\n"; continue; }

        
        for (auto kf : activeMap->keyFrames)
            kfPts.push_back(R_transform_ * kf->t);
    }
    pangolin::CreateWindowAndBind("Full Map Viewer", 1280, 800);
    glEnable(GL_DEPTH_TEST);

    pangolin::OpenGlRenderState rs(
        pangolin::ProjectionMatrix(1280, 800, 500, 500, 640, 400, 0.1f, 10000.f),
        pangolin::ModelViewLookAt(0, -10, -20, 0, 0, 0, 0, -1, 0)
    );
    pangolin::View& d = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, -1280.0f / 800.0f)
        .SetHandler(new pangolin::Handler3D(rs));

    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        d.Activate(rs);

        glPointSize(2.0f);
        glColor3f(0.0f, 0.0f, 0.0f);
        glBegin(GL_POINTS);
        for (auto& p : pts) glVertex3fv(p.data());
        glEnd();

        if (kfPts.size() > 1) {
            glLineWidth(2.0f);
            glColor3f(0.0f, 0.0f, 1.0f);
            glBegin(GL_LINE_STRIP);
            for (auto& p : kfPts) glVertex3fv(p.data());
            glEnd();
        }

        pangolin::FinishFrame();
    }
    
}