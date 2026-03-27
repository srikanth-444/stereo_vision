#include "optimizer.h"


Optimizer::Optimizer(bool verbose){
    optimizer.setVerbose(verbose);
    auto linearSolver = std::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    auto solver_ptr = std::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver));
    auto solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
    optimizer.setAlgorithm(solver);
}
void Optimizer::optimizePose(std::shared_ptr<Frame> frame){
    optimizer.clear();

    Eigen::Matrix3d R = frame->Tcw.block<3,3>(0,0).cast<double>();
    Eigen::Vector3d t = frame->Tcw.block<3,1>(0,3).cast<double>();
    Eigen::Quaterniond qd(R);
    qd.normalize();

    // 2. Add ONLY the Camera Pose Vertex
    auto pose = new g2o::VertexSE3Expmap();
    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(qd,t)); 
    optimizer.addVertex(pose);

    auto landmarks = frame->getLandmarks();
    auto imagePoints = frame->getTrackedPoints();

    std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> edges;
    edges.reserve(landmarks.size());

    for (size_t i = 0; i < landmarks.size(); ++i) {

        auto edge = new g2o::EdgeSE3ProjectXYZOnlyPose();
        
        edge->setVertex(0, pose);
        edge->setMeasurement(imagePoints[i].cast<double>());
        edge->setInformation(Eigen::Matrix2d::Identity());

        // Set the fixed 3D point directly into the edge
        edge->Xw = landmarks[i]->point3D.cast<double>();
        
        // Set camera intrinsics
        edge->fx = frame->intrinsic(0,0);
        edge->fy = frame->intrinsic(1,1);
        edge->cx = frame->intrinsic(0,2);
        edge->cy = frame->intrinsic(1,2);

        // Robust Kernel is still highly recommended for PnP
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        edge->setRobustKernel(rk);
        rk->setDelta(std::sqrt(5.991)); // Chi-square threshold for 2 degrees of freedom

        optimizer.addEdge(edge);
        edges.push_back(edge);
    }

    // 4. Run Optimization
    optimizer.initializeOptimization();
    const int max_iterations = 10; 
    for(int iter = 0; iter < max_iterations; ++iter){
        optimizer.optimize(1); // one iteration at a time

        // Remove high-reprojection-error edges
        for(auto edge : edges){
            if(!edge->level()) { // only consider active edges
                double error = edge->chi2();
                if(error > 5.991){ // same chi2 threshold
                    edge->setLevel(1); // deactivate for next iteration
                }
            }
        }
    }

    Eigen::Matrix4f Tcw=(pose->estimate().to_homogeneous_matrix().cast<float>()).eval();
    // 5. Update Frame Pose
    Eigen::Matrix4f Twr = Tcw.inverse() * frame->extrinsic.inverse();

    Eigen::Matrix3f R_= (Twr.block<3,3>(0,0)).eval();
    Eigen::Vector3f t_= (Twr.block<3,1>(0,3)).eval();
    Eigen::Quaternionf q_(R_);
    q_.normalize();
    frame->setCameraWorldPose(q_,t_);
}