#include "optimizer.h"
#include "utils.h"
#include <unordered_map>


Optimizer::Optimizer(bool verbose){
    this->verbose=verbose;   
}


bool Optimizer::optimizePose(std::shared_ptr<Frame> frame){
    poseoptimizer.clear();
    auto linearSolver = std::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    auto solver_ptr = std::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver));
    auto solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
    poseoptimizer.setAlgorithm(solver);
    std::vector<std::shared_ptr<Frame>> frames;
    Eigen::Quaterniond eqd;
    Eigen::Vector3d    et;
    std::shared_ptr<Frame> cur= frame;
    {   
        std::shared_lock lock(frame->frameMutex);
        eqd = frame->q.cast<double>();
        et  = frame->t.cast<double>();
    }
    while(cur != nullptr){
        std::shared_lock lock(cur->frameMutex);
        frames.push_back(cur);
        cur = cur->rFrame;      
    }
    
    auto* pose = new g2o::VertexSE3Expmap();
    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(eqd, et).inverse());
    poseoptimizer.addVertex(pose);          

    struct EdgeInfo {
        EdgeProjectXYZtoCamera* edge;
        std::shared_ptr<Frame>  cam;     
        int                     landmarkIdx; 
    };
    std::vector<EdgeInfo> edgeInfos;
    int pointVertexId = 1;   // ← unique ids for fixed point vertices

    for(const auto& f : frames){

        struct edgeContext {
            Eigen::Matrix3d mR;
            Eigen::Vector3d mt;
            Eigen::Quaterniond mqd;
            g2o::SE3Quat Tcr_se3;
            std::vector<std::shared_ptr<Landmark>> landmarks;
            std::vector<Eigen::Vector2f> imagePoints;
            std::vector<int> ids;
            Eigen::Matrix3f intrinsic;
        } context;
        {   
            std::shared_lock lock(f->frameMutex);
            context.mR = f->extrinsic.block<3,3>(0,0).cast<double>();
            context.mt = f->extrinsic.block<3,1>(0,3).cast<double>();
            context.mqd = Eigen::Quaterniond(context.mR);
            context.mqd.normalize();
            context.Tcr_se3 = g2o::SE3Quat(context.mqd, context.mt).inverse();
            context.landmarks = f->getLandmarks();
            context.imagePoints = f->getTrackedPoints();
            context.ids = f->getTrackedIds();
            context.intrinsic = f->intrinsic;
            edgeInfos.reserve(edgeInfos.size() + context.landmarks.size());
        }
        
        for(size_t i = 0; i < context.landmarks.size(); ++i){

            if(!context.landmarks[i]) continue;

            std::shared_lock lmLock(context.landmarks[i]->landmarkMutex);
            if(context.landmarks[i]->isBad) continue;

            // fixed point vertex
            auto* vPoint = new g2o::VertexPointXYZ();
            vPoint->setId(pointVertexId++);
            vPoint->setEstimate(context.landmarks[i]->point3D.cast<double>());
            vPoint->setFixed(true);
            poseoptimizer.addVertex(vPoint);

            auto* edge = new EdgeProjectXYZtoCamera();

            Eigen::Matrix<double,2,1> obs;
            obs << context.imagePoints[i][0], context.imagePoints[i][1];

            edge->setVertex(0, vPoint);
            edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                                    poseoptimizer.vertex(0)));
            edge->setMeasurement(obs);
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->fx  = context.intrinsic(0,0);
            edge->fy  = context.intrinsic(1,1);
            edge->cx  = context.intrinsic(0,2);
            edge->cy  = context.intrinsic(1,2);
            edge->Tcr = context.Tcr_se3;

            auto* rk = new g2o::RobustKernelHuber;
            rk->setDelta(std::sqrt(5.991));
            edge->setRobustKernel(rk);

            poseoptimizer.addEdge(edge);
            edgeInfos.push_back({edge, f, static_cast<int>(context.ids[i])});
            edge->computeError();
        }
    }
    
    const int    max_iterations   = 4;
    g2o::SE3Quat best_estimate    = pose->estimate();
    int          best_inlier_count = 0;

    for(int iter = 0; iter < max_iterations; ++iter){

        poseoptimizer.initializeOptimization(0);
        poseoptimizer.optimize(5);

        int active_count = 0;

        for(auto& info : edgeInfos){
            info.edge->computeError();
            double chi2 = info.edge->chi2();

            if(chi2 > 5.991){
                info.edge->setLevel(1);
            } else {
                info.edge->setLevel(0);
                ++active_count;
            }
        }

        if(active_count > best_inlier_count){
            best_inlier_count = active_count;
            best_estimate     = pose->estimate();
        }

        if(active_count < 10){
            std::cout << "Optimization stopped: too few inliers ("
                      << active_count << ")\n";
            break;
        }
    }

    for(const auto& info : edgeInfos){
        if(info.edge->level() == 1){           
            std::unique_lock lock(info.cam->frameMutex);
            info.cam->landmarks[info.landmarkIdx].reset();
        }
    }

    if(best_inlier_count < 10){
        std::cout << "Tracking lost — keeping previous pose\n";
        return false;
    }

    pose->setEstimate(best_estimate);

    Eigen::Matrix4f Twr = pose->estimate()
                              .inverse()
                              .to_homogeneous_matrix()
                              .cast<float>();

    Eigen::Matrix3f R_ = Twr.block<3,3>(0,0);
    Eigen::Vector3f t_ = Twr.block<3,1>(0,3);
    Eigen::Quaternionf q_(R_);
    q_.normalize();
    std::unique_lock lock(frame->frameMutex);
    frame->setCameraWorldPose(q_, t_);
    return true;
}

void Optimizer::localBundleAdjustment(std::shared_ptr<Frame> frame) {
    g2o::SparseOptimizer optimizer;
    auto linearSolver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
    auto solver_ptr = std::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver));
    auto solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
    optimizer.setAlgorithm(solver);
    std::vector<std::shared_ptr<Frame>> frames;
    std::unordered_set<int> uniqueIds;
    std::vector<std::shared_ptr<Landmark>> landmarks;
    std::vector<EdgeProjectXYZtoCamera*> edges;   // ← changed type
    {   
        std::shared_lock lock(frame->frameMutex);
        for(auto& [f, w] : frame->covisibility){
            if(auto spf = f.lock()){
                frames.push_back(spf);
            }
        }
        frames.push_back(frame);
    }
    
    for (auto& f : frames) {
        std::shared_lock lock(f->frameMutex);
        for (auto& lm : f->getLandmarks()) {
            if (uniqueIds.insert(lm->id).second) {
                landmarks.push_back(lm);
            }
        }
    }

    const int LANDMARK_ID_OFFSET = 1000000;
    edges.reserve(landmarks.size());

    for (auto lm : landmarks) {
        
        auto v = new g2o::VertexPointXYZ();
        std::unordered_map<std::weak_ptr<Frame>, int, WeakPtrHash, WeakPtrEqual> observations;
        int lmid;
        {   
            std::shared_lock lock(lm->landmarkMutex);
            if(lm->isBad) continue;
            v->setId(lm->id + LANDMARK_ID_OFFSET);  
            v->setEstimate(lm->point3D.cast<double>());
            v->setFixed(false);
            v->setMarginalized(true);
            lmid = lm->id;
            observations = lm->observations;
        }
        optimizer.addVertex(v);

        for (const auto& [weakFrame, fid] : observations) {
            auto mframe = weakFrame.lock();
            if (!mframe) continue;
            std::shared_lock lock(mframe->frameMutex);
            if (!optimizer.vertex(mframe->id)) {
                auto vertex = new g2o::VertexSE3Expmap();
                vertex->setId(mframe->id);

                Eigen::Matrix3d R = mframe->Tcw.block<3,3>(0,0).cast<double>();
                Eigen::Vector3d t = mframe->Tcw.block<3,1>(0,3).cast<double>();
                Eigen::Quaterniond qd(R);
                qd.normalize();
            
                vertex->setEstimate(g2o::SE3Quat(qd, t)); 

                bool isLocal = std::find(frames.begin(), frames.end(), mframe) != frames.end();
                vertex->setFixed(!isLocal);
                optimizer.addVertex(vertex);  
            }

            auto* landmarkVertex = optimizer.vertex(lmid + LANDMARK_ID_OFFSET);
            auto* frameVertex    = optimizer.vertex(mframe->id);
            if (!landmarkVertex || !frameVertex) continue;

            // compute Tcr for this camera
            Eigen::Matrix3d Rcr = mframe->extrinsic.block<3,3>(0,0).cast<double>();
            Eigen::Vector3d tcr = mframe->extrinsic.block<3,1>(0,3).cast<double>();
            Eigen::Quaterniond qcr(Rcr);
            qcr.normalize();

            auto* edge = new EdgeProjectXYZtoCamera();   // ← changed
            Eigen::Matrix<double, 2, 1> obs;
            obs << mframe->keyPoints[fid].pt.x, mframe->keyPoints[fid].pt.y;

            edge->setVertex(0, landmarkVertex);
            edge->setVertex(1, frameVertex);
            edge->setMeasurement(obs);
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->fx  = mframe->intrinsic(0, 0);
            edge->fy  = mframe->intrinsic(1, 1);
            edge->cx  = mframe->intrinsic(0, 2);
            edge->cy  = mframe->intrinsic(1, 2);
            edge->Tcr = g2o::SE3Quat(qcr, tcr).inverse();   // ← added

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
    int bad = 0;
    for(auto* edge : edges){
        edge->computeError();
        double error = edge->chi2();

        auto* v = static_cast<g2o::VertexPointXYZ*>(edge->vertex(0));
        int lm_id = v->id() - LANDMARK_ID_OFFSET;

        landmarkError[lm_id] += error;
        landmarkObsCount[lm_id]++;
    }

    for(auto& lm : landmarks){
        std::unique_lock lock(lm->landmarkMutex);
        if(lm->isBad) continue;
        int id = lm->id;
        double avgError = landmarkError[id] / landmarkObsCount[id];
        auto age = lm->getAge(frame->id);
        if(age >= 10){
            if((avgError > 5.991) || landmarkObsCount[id] < 3){
                lm->setBadFlag();
                bad++;
                for(auto [frame, fid] : lm->observations){
                    if(auto f = frame.lock()){
                        std::unique_lock lock(f->frameMutex);
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
                        std::unique_lock lock(f->frameMutex);
                        f->landmarks[fid].reset();
                    }
                }
                lm->erase();
            }
        }
    }

    for (auto& lm : landmarks) {
        std::unique_lock lock(lm->landmarkMutex);
        if (lm->isBad) continue;
        auto* v = static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(lm->id + LANDMARK_ID_OFFSET));
        if (!v) continue;
        lm->point3D = v->estimate().cast<float>();
    }

    for (auto& f : frames) {
        auto* v = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(f->id));
        if (!v) continue;

        Eigen::Matrix4f Tcw = v->estimate().to_homogeneous_matrix().cast<float>();
        Eigen::Matrix3f Rcw = Tcw.block<3,3>(0,0);
        Eigen::Vector3f tcw = Tcw.block<3,1>(0,3);
        Eigen::Matrix3f Rwc = Rcw.transpose();
        Eigen::Vector3f twc = -Rwc * tcw;
        Eigen::Matrix4f Twc = Eigen::Matrix4f::Identity();
        Twc.block<3,3>(0,0) = Rwc;
        Twc.block<3,1>(0,3) = twc;

        Eigen::Matrix4f Twr = Twc * f->extrinsic.inverse();

        Eigen::Matrix3f R = Twr.block<3,3>(0,0);
        Eigen::Vector3f t = Twr.block<3,1>(0,3);
        Eigen::Quaternionf q(R);
        std::unique_lock lock(f->frameMutex);
        f->setCameraWorldPose(q, t);
    }
}
