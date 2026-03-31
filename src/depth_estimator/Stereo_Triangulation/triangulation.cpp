#include "triangulation.h"
#include <iostream>

Stereo::Stereo(Eigen::Matrix3f left_intrinsic,Eigen::Matrix4f left_extrinsic,Eigen::Vector4f left_dist,
                Eigen::Matrix3f right_intrinsic,Eigen::Matrix4f right_extrinsic,Eigen::Vector4f right_dist, int W, int H):
                l_K(left_intrinsic),l_extrinsic(left_extrinsic),l_dist(left_dist),
                r_K(right_intrinsic),r_extrinsic(right_extrinsic),r_dist(right_dist)
                {

                    Eigen::Matrix4f T_rel = r_extrinsic.inverse() * l_extrinsic;
                    this->R_rel = T_rel.block<3,3>(0,0);
                    this->t_rel = T_rel.block<3,1>(0,3);
                

                    // std::cout<<"left intrinsic: "<<l_K
                    //             <<"distrotion: "<<l_dist<<std::endl;
                    // std::cout<<"right intrinsic: "<<r_K
                    //             <<"distrotion: "<<r_dist<<std::endl;
                    // std::cout<<"relative R: "<<R_rel
                    //             <<"relative t: "<<t_rel<<std::endl;
                
                
                cv::eigen2cv(l_K, cv_l_K);
                cv::eigen2cv(r_K, cv_r_K);
                cv::eigen2cv(R_rel, cv_R_rel);
                cv::eigen2cv(t_rel, cv_T_rel);
                cv::eigen2cv(l_dist,cv_l_dist);
                cv::eigen2cv(r_dist,cv_r_dist);
                cv_l_K.convertTo(cv_l_K, CV_64F);
                cv_r_K.convertTo(cv_r_K, CV_64F);
                cv_R_rel.convertTo(cv_R_rel, CV_64F);
                cv_T_rel.convertTo(cv_T_rel, CV_64F);
                cv_l_dist.convertTo(cv_l_dist, CV_64F);
                cv_r_dist.convertTo(cv_r_dist, CV_64F);

                // std::cout<<"left intrinsic: "<<cv_l_K
                //             <<"distrotion: "<<cv_l_dist<<std::endl;
                // std::cout<<"right intrinsic: "<<cv_r_K
                //             <<"distrotion: "<<cv_r_dist<<std::endl;
                // std::cout<<"relative R: "<<cv_R_rel
                //             <<"relative t: "<<cv_T_rel<<std::endl;

                // 3. Stereo Rectification
                cv::stereoRectify(cv_l_K, cv_l_dist, cv_r_K, cv_r_dist, cv::Size(W, H), cv_R_rel, cv_T_rel, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 0);
                // std::cout<<"R1: "<<R1
                //             <<"R2: "<<R2
                //             <<"P1: "<<P2
                //             <<"P2: "<<P2<<std::endl;
            
                fx = P1.at<double>(0,0);
                fy = P1.at<double>(1,1);
                cx = P1.at<double>(0,2);
                cy = P1.at<double>(1,2);
                baseline = -P2.at<double>(0,3) / P2.at<double>(0,0); 
                }
std::vector<cv::Point2f> eigenToCvPoints(const Eigen::MatrixXf& mat) {
    std::vector<cv::Point2f> pts;
    pts.reserve(mat.rows());

    for (int i = 0; i < mat.rows(); ++i) {
        pts.emplace_back(mat(i, 0), mat(i, 1));
    }

    return pts;
}


void Stereo::getDepth(std::shared_ptr<Frame> &leftFrame,std::shared_ptr<Frame> &rightFrame,float epipolar_threshold,std::vector<Eigen::Vector3f>& points_3d,std::vector<int>& featureIds)
{   
    auto pts_l = leftFrame->getNotAssociatedPoints();
    auto pts_r = rightFrame->getNotAssociatedPoints();
    auto des_l = leftFrame->getNotAssociatedDescriptors();
    auto des_r = rightFrame->getNotAssociatedDescriptors();
    auto left_ids=leftFrame->getNotAssociatedIndices(); 
    auto right_ids=rightFrame->getNotAssociatedIndices();

    // std::cout<<"points: "<<pts_l(0)<<pts_r(0)<<std::endl;
    points_3d.reserve(left_ids.size());
    featureIds.reserve(left_ids.size());

    auto rect_l=eigenToCvPoints(pts_l);
    auto rect_r=eigenToCvPoints(pts_r);
    // std::cout<<"points: "<<rect_l[0]<<rect_r[0]<<std::endl;
    std::map<int, std::vector<int>> row_to_right_indices;
    for (size_t i = 0; i < rect_r.size(); ++i) {
        int row = static_cast<int>(std::round(rect_r[i].y));
        row_to_right_indices[row].push_back((int)i);
    }

    std::vector<std::pair<int,int>> final_matches;
    std::set<int> used_right_indices;

    for (size_t i = 0; i < rect_l.size(); ++i) {
        int row = static_cast<int>(std::round(rect_l[i].y));

        std::vector<int> candidates;
        for (int r = row - (int)epipolar_threshold; r <= row + (int)epipolar_threshold; ++r) {
            auto it = row_to_right_indices.find(r);
            if (it != row_to_right_indices.end()) {
                candidates.insert(candidates.end(), it->second.begin(), it->second.end());
            }
        }

        if (candidates.empty()) continue;
        // std::cout<<"leftdescriptors: "<<des_l
        //             <<"rightdescriptors: "<<des_r<<std::endl;
        // std::cout << "candidates: ";
        // for (int c : candidates) {
        //     std::cout << c << " ";
        // }
        // std::cout << std::endl;
        int best_idx = leftFrame->featureExtractor->match(des_l.row((int)i), des_r, candidates, 0.7);
        // std::cout<<best_idx<<std::endl;
        if (best_idx < 0) continue;
        if (used_right_indices.count(best_idx)) continue;
        if (rect_l[i].x <= rect_r[best_idx].x) continue;
        float disparity = rect_l[i].x - rect_r[best_idx].x;
        if (disparity < 1.0f) continue;

        float Z = fx * baseline / disparity;
        if(Z>15)continue;
        float X = (rect_l[i].x - cx) * Z / fx;
        float Y = (rect_l[i].y - cy) * Z / fy;

        // std::cout<<"X: "<<X
        //          <<"Y: "<<Y
        //          <<"Z: "<<Z<<std::endl;
        // project
        float u_r_proj = fx * X / Z + cx;
        float v_r_proj = fy * Y / Z + cy;
        // std::cout<<u_r_proj<<","<<v_r_proj<<std::endl;
        float reproj_error = std::sqrt(
            std::pow(u_r_proj - rect_l[i].x, 2) +
            std::pow(v_r_proj - rect_l[i].y, 2)
        );
        // std::cout<<reproj_error<<std::endl;
        if (reproj_error > 3.0f) continue;
        
        auto p_cam=Eigen::Vector3f(X,Y,Z);
        Eigen::Vector3f p_world = l_extrinsic.block<3,3>(0,0) * p_cam 
                          + l_extrinsic.block<3,1>(0,3);
        points_3d.emplace_back(p_world);
        featureIds.emplace_back(left_ids[i]);
        final_matches.push_back(std::make_pair(i, best_idx));
    }
    //visualizeMatchesBlackCanvas(rect_l,rect_r,final_matches,leftFrame->image.cols,leftFrame->image.rows);

}
void Stereo::visualizeMatchesBlackCanvas(
    const std::vector<cv::Point2f>& rect_l,
    const std::vector<cv::Point2f>& rect_r,
    const std::vector<std::pair<int,int>>& matches,
    int canvas_width,
    int canvas_height)
{
    // Create black canvas
    cv::Mat canvas = cv::Mat::zeros(canvas_height, canvas_width, CV_8UC3);

    for(auto pts:rect_l){
        // Left point in blue
        cv::circle(canvas, pts, 3, cv::Scalar(255,0,0), -1);
    }
    for(auto pts:rect_r){
        // Right point in red
        cv::circle(canvas, pts, 3, cv::Scalar(0,0,255), -1);
    }

    // Draw matches
    for (size_t i = 0; i < matches.size(); ++i)
    {
        int l_idx = matches[i].first;
        int r_idx = matches[i].second;
        // Optional: draw green line connecting them
        cv::line(canvas, rect_l[l_idx], rect_r[r_idx], cv::Scalar(0,255,0), 1);
    }

    // Show result
    cv::imshow("Stereo Matches on Black Canvas", canvas);
    cv::waitKey(0);
}

void Stereo::rectifyPoints(std::shared_ptr<Frame> &leftFrame,
    std::shared_ptr<Frame> &rightFrame){


        std::vector<cv::Point2f> pts_l, pts_r;
        cv::KeyPoint::convert(leftFrame->keyPoints, pts_l);
        cv::KeyPoint::convert(rightFrame->keyPoints, pts_r);
        
        std::vector<cv::Point2f> rect_l;
        std::vector<cv::Point2f> rect_r; 
        cv::undistortPoints(pts_l, rect_l, cv_l_K, cv_l_dist, R1, P1);
        cv::undistortPoints(pts_r, rect_r, cv_r_K, cv_r_dist, R2, P2);
        // std::cout<<rect_l[0]<<rect_r[0]<<std::endl;
        cv::Mat K1_new = P1(cv::Rect(0, 0, 3, 3));
        cv::Mat K2_new = P2(cv::Rect(0, 0, 3, 3));
        Eigen::Matrix3f K1_eigen, K2_eigen;
        cv::cv2eigen(K1_new, K1_eigen);
        cv::cv2eigen(K2_new, K2_eigen);

        leftFrame->intrinsic = K1_eigen;
        rightFrame->intrinsic = K2_eigen;

        for (size_t i = 0; i < leftFrame->keyPoints.size(); i++) {
            leftFrame->keyPoints[i].pt = rect_l[i];
        }

        for (size_t i = 0; i < rightFrame->keyPoints.size(); i++) {
            rightFrame->keyPoints[i].pt = rect_r[i];
        }
    }

void Stereo::rectifyImage(std::shared_ptr<Frame> &frame){
    cv::Mat undistorted;
    cv::Mat K,distCoeffs;
    cv::eigen2cv(frame->intrinsic,K);
    cv::eigen2cv(frame->dist_coefficents,distCoeffs);
    cv::undistort(frame->image, undistorted, K, distCoeffs);
    frame->image = undistorted;
}
