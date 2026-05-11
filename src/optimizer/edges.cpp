#include "edges.h"
#include "vertex.h"
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/se3_ops.h>

EdgeProjectXYZtoCamera::EdgeProjectXYZtoCamera(){}

void EdgeProjectXYZtoCamera::computeError(){
    const auto* vPoint = static_cast<const g2o::VertexPointXYZ*>(_vertices[0]);
    const auto* vPose  = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
    Eigen::Vector2d obs(_measurement);
    _error = obs - cProject((Tcr * vPose->estimate()).map(vPoint->estimate()));
}

bool EdgeProjectXYZtoCamera::isDepthPositive(){
    const auto* vPoint = static_cast<const g2o::VertexPointXYZ*>(_vertices[0]);
    const auto* vPose  = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
    return (Tcr * vPose->estimate()).map(vPoint->estimate())(2) > 0;
}

void EdgeProjectXYZtoCamera::linearizeOplus() {
    const auto* vPoint = static_cast<const g2o::VertexPointXYZ*>(_vertices[0]);
    const auto* vPose  = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);

    Eigen::Vector3d Xw  = vPoint->estimate();
    g2o::SE3Quat    Trw = vPose->estimate();
    Eigen::Vector3d p_r = Trw.map(Xw);
    Eigen::Vector3d p_c = Tcr.map(p_r);

    if (p_c[2] < 0.01) {
        _jacobianOplusXi.setZero();   // w.r.t. point
        _jacobianOplusXj.setZero();   // w.r.t. pose
        return;
    }

    double x     = p_c[0];
    double y     = p_c[1];
    double invz  = 1.0 / p_c[2];
    double invz2 = invz * invz;

    Eigen::Matrix3d Rcr = Tcr.rotation().toRotationMatrix();
    Eigen::Matrix3d Rrw = Trw.rotation().toRotationMatrix();

    Eigen::Matrix<double,2,3> J_pi;
    J_pi(0,0) =  fx * invz;
    J_pi(0,1) =  0;
    J_pi(0,2) = -fx * x * invz2;
    J_pi(1,0) =  0;
    J_pi(1,1) =  fy * invz;
    J_pi(1,2) = -fy * y * invz2;

    // w.r.t. point (new)
    _jacobianOplusXi = -J_pi * Rcr * Rrw;

    // w.r.t. pose (unchanged)
    Eigen::Matrix<double,3,6> dp_r;
    dp_r <<        0,  p_r[2], -p_r[1], 1, 0, 0,
            -p_r[2],       0,  p_r[0], 0, 1, 0,
             p_r[1], -p_r[0],       0, 0, 0, 1;

    _jacobianOplusXj = -J_pi * Rcr * dp_r;
}

bool EdgeProjectXYZtoCamera::read(std::istream& is) {
    g2o::internal::readVector(is, _measurement);
    return readInformationMatrix(is);
}

bool EdgeProjectXYZtoCamera::write(std::ostream& os) const {
    g2o::internal::writeVector(os, measurement());
    return writeInformationMatrix(os);
}

Eigen::Vector2d EdgeProjectXYZtoCamera::cProject(Eigen::Vector3d trans_xyz){
    const double invz = 1.0 / trans_xyz[2];
    Eigen::Vector2d res;
    res[0] = trans_xyz[0] * invz * fx + cx;
    res[1] = trans_xyz[1] * invz * fy + cy;
    return res;
}



// EdgeInertial::EdgeInertial(){};
    

// void EdgeInertial::computeError()
// {
//     // retrieve vertices
//     const g2o::VertexSE3Expmap* VP1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
//     const VertexVelocity*       VV1 = static_cast<const VertexVelocity*>(_vertices[1]);
//     const g2o::VertexSE3Expmap* VP2 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[4]);
//     const VertexVelocity*       VV2 = static_cast<const VertexVelocity*>(_vertices[5]);
//     const VertexGyroBias*      VG  = static_cast<const VertexGyroBias*>(_vertices[2]);
//     const VertexAccBias*       VA  = static_cast<const VertexAccBias*>(_vertices[3]);

//     // extract states
//     const Eigen::Matrix3d Ri = VP1->estimate().rotation().toRotationMatrix();
//     const Eigen::Vector3d pi = VP1->estimate().translation();
//     const Eigen::Vector3d vi = VV1->estimate();
//     const Eigen::Matrix3d Rj = VP2->estimate().rotation().toRotationMatrix();
//     const Eigen::Vector3d pj = VP2->estimate().translation();
//     const Eigen::Vector3d vj = VV2->estimate();

//      // bias deltas — how much has bias changed since preintegration was computed
//     const Eigen::Vector3d dbg = VG->estimate() - bg0;   
//     const Eigen::Vector3d dba = VA->estimate() - ba0;   

//     // first order bias correction  (the Jacobians from your previous question)
//     const Eigen::Matrix3d dR_corrected = dR * ExpSO3(JRg * dbg);
//     const Eigen::Vector3d dV_corrected = dV + JVg * dbg + JVa * dba;
//     const Eigen::Vector3d dP_corrected = dP + JPg * dbg + JPa * dba;

//     // compute residuals
//     const Eigen::Vector3d er = LogSO3(dR_corrected.transpose() * Ri.transpose() * Rj);
//     const Eigen::Vector3d ev = Ri.transpose() * (vj - vi - g * dt) - dV_corrected;
//     const Eigen::Vector3d ep = Ri.transpose() * (pj - pi - vi * dt - 0.5 * g * dt * dt) - dP_corrected;

//     _error << er, ev, ep;
// }

// void EdgeInertial::linearizeOplus()
// {
//     const g2o::VertexSE3Expmap* VP1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
//     const VertexVelocity*       VV1 = static_cast<const VertexVelocity*>(_vertices[1]);
//     const VertexGyroBias*       VG  = static_cast<const VertexGyroBias*>(_vertices[2]);
//     const VertexAccBias*        VA  = static_cast<const VertexAccBias*>(_vertices[3]);
//     const g2o::VertexSE3Expmap* VP2 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[4]);
//     const VertexVelocity*       VV2 = static_cast<const VertexVelocity*>(_vertices[5]);

//     // extract states
//     const Eigen::Matrix3d Ri = VP1->estimate().rotation().toRotationMatrix();
//     const Eigen::Vector3d pi = VP1->estimate().translation();
//     const Eigen::Vector3d vi = VV1->estimate();
//     const Eigen::Matrix3d Rj = VP2->estimate().rotation().toRotationMatrix();
//     const Eigen::Vector3d pj = VP2->estimate().translation();
//     const Eigen::Vector3d vj = VV2->estimate();

//     // bias deltas
//     const Eigen::Vector3d dbg = VG->estimate() - bg0;
//     const Eigen::Vector3d dba = VA->estimate() - ba0;

//     // bias corrected measurements
//     const Eigen::Matrix3d dRc = dR * ExpSO3(JRg * dbg);
//     const Eigen::Vector3d dVc = dV + JVg * dbg + JVa * dba;
//     const Eigen::Vector3d dPc = dP + JPg * dbg + JPa * dba;

//     // frequently used quantities
//     const Eigen::Matrix3d RiT  = Ri.transpose();
//     const Eigen::Matrix3d RjT  = Rj.transpose();
//     const Eigen::Vector3d eR   = _error.block<3,1>(0,0);  // rotation residual
//     const Eigen::Matrix3d JrInv = InverseRightJacobianSO3(eR);

//     // relative rotation between keyframes
//     const Eigen::Matrix3d eRmat = dRc.transpose() * RiT * Rj;

//     // shorthand for common terms
//     const Eigen::Vector3d dp  = pj - pi - vi * dt - 0.5 * g * dt * dt;
//     const Eigen::Vector3d dv  = vj - vi - g * dt;

//     _jacobianOplus[0].setZero();
//     _jacobianOplus[0].block<3,3>(0,0) = -JrInv * RjT * Ri;
//     _jacobianOplus[0].block<3,3>(3,0) =  skew(RiT * dv);
//     _jacobianOplus[0].block<3,3>(6,0) =  skew(RiT * dp);
//     _jacobianOplus[0].block<3,3>(6,3) = -RiT;
//     _jacobianOplus[1].setZero();
//     _jacobianOplus[1].block<3,3>(3,0) = -RiT;
//     _jacobianOplus[1].block<3,3>(6,0) = -RiT * dt;
//     _jacobianOplus[2].setZero();
//     _jacobianOplus[2].block<3,3>(0,0) = -JrInv * eRmat.transpose() * RightJacobianSO3(JRg * dbg) * JRg;
//     _jacobianOplus[2].block<3,3>(3,0) = -JVg;
//     _jacobianOplus[2].block<3,3>(6,0) = -JPg;
//     _jacobianOplus[3].setZero();
//     _jacobianOplus[3].block<3,3>(3,0) = -JVa;
//     _jacobianOplus[3].block<3,3>(6,0) = -JPa;
//     _jacobianOplus[4].setZero();
//     _jacobianOplus[4].block<3,3>(0,0) = JrInv;
//     _jacobianOplus[4].block<3,3>(6,3) = RiT;
//     _jacobianOplus[5].setZero();
//     _jacobianOplus[5].block<3,3>(3,0) = RiT;
// }

// Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R)
// {
//     const double tr = R(0,0)+R(1,1)+R(2,2);
//     Eigen::Vector3d w;
//     w << (R(2,1)-R(1,2))/2, (R(0,2)-R(2,0))/2, (R(1,0)-R(0,1))/2;
//     const double costheta = (tr-1.0)*0.5f;
//     if(costheta>1 || costheta<-1)
//         return w;
//     const double theta = acos(costheta);
//     const double s = sin(theta);
//     if(fabs(s)<1e-5)
//         return w;
//     else
//         return theta*w/s;
// }

// Eigen::Matrix3d ExpSO3(const Eigen::Vector3d& phi)
// {
//     const double angle = phi.norm();
//     if(angle < 1e-5)
//         return Eigen::Matrix3d::Identity();

//     const Eigen::Vector3d axis = phi / angle;
//     const double s = sin(angle);
//     const double c = cos(angle);
//     Eigen::Matrix3d K;
//     K <<        0, -axis.z(),  axis.y(),
//           axis.z(),        0, -axis.x(),
//          -axis.y(),  axis.x(),        0;

//     return Eigen::Matrix3d::Identity() + s * K + (1.0 - c) * K * K;
// }

// Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d& phi)
// {
//     const double angle = phi.norm();
//     if(angle < 1e-5)
//         return Eigen::Matrix3d::Identity();

//     Eigen::Matrix3d K;
//     const Eigen::Vector3d axis = phi / angle;
//     K <<        0, -axis.z(),  axis.y(),
//           axis.z(),        0, -axis.x(),
//          -axis.y(),  axis.x(),        0;

//     return Eigen::Matrix3d::Identity()
//            - ((1.0 - cos(angle)) / angle) * K
//            + ((angle - sin(angle)) / angle) * K * K;
// }

// Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d& phi)
// {
//     const double angle = phi.norm();
//     if(angle < 1e-5)
//         return Eigen::Matrix3d::Identity();

//     Eigen::Matrix3d K;
//     const Eigen::Vector3d axis = phi / angle;
//     K <<        0, -axis.z(),  axis.y(),
//           axis.z(),        0, -axis.x(),
//          -axis.y(),  axis.x(),        0;

//     return Eigen::Matrix3d::Identity()
//            + 0.5 * angle * K
//            + (1.0 - (angle * 0.5 * cos(angle * 0.5)) / sin(angle * 0.5)) * K * K;
// }

// Eigen::Matrix3d skew(const Eigen::Vector3d& v)
// {
//     Eigen::Matrix3d S;
//     S <<  0,    -v.z(),  v.y(),
//           v.z(),  0,    -v.x(),
//          -v.y(),  v.x(),  0;
//     return S;
// }