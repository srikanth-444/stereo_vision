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

    // std::cout << "landmarks: " << landmarks.size()
    //       << " imagePoints: " << imagePoints.size() << std::endl;
    std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> edges;
    edges.reserve(landmarks.size());
    // std::cout<<"edges pointer created"<<std::endl;
    for (size_t i = 0; i < landmarks.size(); ++i) {
        // std::cout<<"iteration: "<<i
        //         <<"landmark id: "<<landmarks[i]->id
        //         <<"point3d: "<<landmarks[i]->point3D
        //         <<"image points"<<imagePoints[i]<<std::endl;
        auto edge = new g2o::EdgeSE3ProjectXYZOnlyPose();
        auto variance = static_cast<double>(landmarks[i]->point3D(2));
        double min_variance = 1e-6; // or some small epsilon
        variance = std::max(variance, min_variance);
        Eigen::Matrix<double,2,1> obs;
        obs << imagePoints[i][0], imagePoints[i][1];
        edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        edge->setMeasurement(obs);
        edge->setInformation(Eigen::Matrix2d::Identity()/variance);

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
        edges.emplace_back(edge);
    }
    // std::cout<<"intializing optimization"<<std::endl;
    // 4. Run Optimization
    optimizer.initializeOptimization();
    const int max_iterations = 4; 
    for(int iter = 0; iter < max_iterations; ++iter){
        // std::cout<<"starting iter"<<std::endl;
        optimizer.optimize(10); // one iteration at a time

        // Remove high-reprojection-error edges
        for(auto edge : edges){
            if(!edge->level()) { // only consider active edges
                double error = edge->chi2();
                if(error > 5.991){ // same chi2 threshold
                    edge->setLevel(1); // deactivate for next iteration
                }
                else {
                    edge->setLevel(0);
                    edge->setRobustKernel(0); // Only remove Huber for inliers
                }
            }
            // if(iter == 2)edge->setRobustKernel(0);
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