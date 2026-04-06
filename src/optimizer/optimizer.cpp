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
    auto ids=frame->getTrackedIds();
    std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> edges;
    edges.reserve(landmarks.size());

    for (size_t i = 0; i < landmarks.size(); ++i) {
        auto edge = new g2o::EdgeSE3ProjectXYZOnlyPose();
        auto variance = static_cast<double>(landmarks[i]->point3D(2));
        double min_variance = 1e-6; 
        variance = std::max(variance, min_variance);
        Eigen::Matrix<double,2,1> obs;
        obs << imagePoints[i][0], imagePoints[i][1];
        edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        edge->setMeasurement(obs);
        edge->setInformation(Eigen::Matrix2d::Identity());

        edge->Xw = landmarks[i]->point3D.cast<double>();
        

        edge->fx = frame->intrinsic(0,0);
        edge->fy = frame->intrinsic(1,1);
        edge->cx = frame->intrinsic(0,2);
        edge->cy = frame->intrinsic(1,2);

        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        edge->setRobustKernel(rk);
        rk->setDelta(std::sqrt(5.991));

        optimizer.addEdge(edge);
        edges.emplace_back(edge);
    }
    
    // 4. Run Optimization
    const int max_iterations = 4; 
    for(int iter = 0; iter < max_iterations; ++iter){

        optimizer.initializeOptimization(0);
         // 2. Run a few iterations
        optimizer.optimize(5); 

        int active_count = 0;

        // 3. Update ALL edges (both active and inactive)
        for(auto* edge : edges) {
            // Calculate error based on current vertex positions
                edge->computeError(); 
                double error = edge->chi2();

                if(error > 5.991) { 
                    edge->setLevel(1); // Deactivate/Keep inactive
                }
                else {
                    edge->setLevel(0); // Re-activate/Keep active
                    active_count++;
                }
            }

            // 4. Safety check: stop if the graph becomes too thin
            if(active_count < 10) {
                std::cout << "Optimization stopped: too few inliers." << std::endl;
                break;
            }
    }

    for(int i=0; i<edges.size(); i++){
        if(edges[i]->level() == 1){

            frame->landmarks[ids[i]].reset();
    
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

void Optimizer::localBundleAdjustment(std::shared_ptr<Frame> frame, std::shared_ptr<Map> currentMap) {
    auto linearSolver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
    auto solver_ptr = std::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver));
    auto solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);
    auto frames = currentMap->getClosestKeyFrames(frame);

    // Collect unique landmarks
    std::unordered_set<int> uniqueIds;
    std::vector<std::shared_ptr<Landmark>> landmarks;
    for (auto& f : frames) {
        for (auto& lm : f->getLandmarks()) {
            if (uniqueIds.insert(lm->id).second) {
                landmarks.push_back(lm);
            }
        }
    }

    const int LANDMARK_ID_OFFSET = 1000000;

    std::vector<g2o::EdgeSE3ProjectXYZ*> edges;
    edges.reserve(landmarks.size());

    for (auto lm : landmarks) {
        
        auto v = new g2o::VertexPointXYZ();
        v->setId(lm->id + LANDMARK_ID_OFFSET);  
        v->setEstimate(lm->point3D.cast<double>());
        v->setFixed(false);
        v->setMarginalized(true);
        optimizer.addVertex(v);

        for (const auto& [weakFrame, fid] : lm->observations) {
            auto mframe = weakFrame.lock();
            if (!mframe) continue;

            if (!optimizer.vertex(mframe->id)) {
                auto vertex = new g2o::VertexSE3Expmap();
                vertex->setId(mframe->id);

                Eigen::Matrix3d R = mframe->Tcw.block<3,3>(0,0).cast<double>();
                Eigen::Vector3d t = mframe->Tcw.block<3,1>(0,3).cast<double>();
                Eigen::Quaterniond qd(R);
                qd.normalize();
            
                vertex->setEstimate(g2o::SE3Quat(qd,t)); 

                bool isLocal = std::find(frames.begin(), frames.end(), mframe) != frames.end();
                vertex->setFixed(!isLocal);
                optimizer.addVertex(vertex);  
            }


            auto* landmarkVertex = optimizer.vertex(lm->id + LANDMARK_ID_OFFSET); // FIX 1
            auto* frameVertex    = optimizer.vertex(mframe->id);
            if (!landmarkVertex || !frameVertex) continue;

            auto edge = new g2o::EdgeSE3ProjectXYZ();
            Eigen::Matrix<double, 2, 1> obs;
            obs << mframe->keyPoints[fid].pt.x, mframe->keyPoints[fid].pt.y;

            edge->setVertex(0, landmarkVertex);
            edge->setVertex(1, frameVertex);
            edge->setMeasurement(obs);
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->fx = mframe->intrinsic(0, 0);
            edge->fy = mframe->intrinsic(1, 1);
            edge->cx = mframe->intrinsic(0, 2);
            edge->cy = mframe->intrinsic(1, 2);

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            edge->setRobustKernel(rk);
            rk->setDelta(std::sqrt(5.991));

            optimizer.addEdge(edge);
            edges.emplace_back(edge);
        }
    }


    const int max_iterations = 4; 
    for(int iter = 0; iter < max_iterations; ++iter){

        optimizer.initializeOptimization(0);
    
        optimizer.optimize(5); 

        int active_count = 0;

        for(auto* edge : edges) {
                edge->computeError(); 
                double error = edge->chi2();

                if(error > 5.991) { 
                    edge->setLevel(1); 
                }
                else {
                    edge->setLevel(0); 
                    active_count++;
                }
            }

            if(active_count < 10) {
                std::cout << "Optimization stopped: too few inliers." << std::endl;
                break;
            }
    }

    std::unordered_map<int, double> landmarkError;
    std::unordered_map<int, int> landmarkObsCount;
    int bad=0;
    for(auto* edge : edges){
        edge->computeError();
        double error = edge->chi2();

        auto* v = static_cast<g2o::VertexPointXYZ*>(edge->vertex(0));
        int lm_id = v->id() - LANDMARK_ID_OFFSET;

        landmarkError[lm_id] += error;
        landmarkObsCount[lm_id]++;
    }
    for(auto& lm : landmarks){
        int id = lm->id;

        double avgError = landmarkError[id] / landmarkObsCount[id];
        auto age= lm->getAge(frame->id);
        if(age>6){
            if((avgError > 5.991)||landmarkObsCount[id]<3){
                lm->setBadFlag();
                bad++;
                for(auto [frame, fid] : lm->observations){
                    if(auto f = frame.lock()){
                        f->landmarks[fid].reset();
                    }
                }
                lm->erase();
            }
        }
        else{
            if(avgError > 5.991){
                lm->setBadFlag();
                bad++;
                for(auto [frame, fid] : lm->observations){
                    if(auto f = frame.lock()){
                        f->landmarks[fid].reset();
                    }
                }
                lm->erase();
            }
        }
    }


}

