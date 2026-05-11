#ifndef EDGES_H
#define EDGES_H
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_multi_edge.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

class EdgeProjectXYZtoCamera: public g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexPointXYZ, g2o::VertexSE3Expmap>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    g2o::SE3Quat Tcr;
    double fx, fy, cx, cy, bf;

    bool read(std::istream& is);
    bool write(std::ostream& os) const;
    void computeError();
    bool isDepthPositive();

    Eigen::Vector2d cProject(Eigen::Vector3d xyz);
    EdgeProjectXYZtoCamera();

    virtual void linearizeOplus();
};

// class EdgeInertial : public g2o::BaseMultiEdge<9, Eigen::Matrix<double,9,1>>
// {
// public:
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//     EdgeInertial();

//     virtual void computeError() override;
//     virtual void linearizeOplus() override;


//     Eigen::Matrix3d dR;
//     Eigen::Vector3d dV;
//     Eigen::Vector3d dP;

//     Eigen::Matrix3d JRg;
//     Eigen::Matrix3d JVg;
//     Eigen::Matrix3d JVa;
//     Eigen::Matrix3d JPg;
//     Eigen::Matrix3d JPa;


//     Eigen::Vector3d bg0;
//     Eigen::Vector3d ba0;


//     Eigen::Vector3d g;
//     double dt;



//     void setPreintegrated(
//         const Eigen::Matrix3d& dR_,
//         const Eigen::Vector3d& dV_,
//         const Eigen::Vector3d& dP_,
//         const Eigen::Matrix3d& JRg_,
//         const Eigen::Matrix3d& JVg_,
//         const Eigen::Matrix3d& JVa_,
//         const Eigen::Matrix3d& JPg_,
//         const Eigen::Matrix3d& JPa_,
//         const Eigen::Vector3d& bg0_,
//         const Eigen::Vector3d& ba0_,
//         const Eigen::Vector3d& g_,
//         const double dt_)
//     {
//         dR  = dR_;   dV  = dV_;   dP  = dP_;
//         JRg = JRg_;  JVg = JVg_;  JVa = JVa_;
//         JPg = JPg_;  JPa = JPa_;
//         bg0 = bg0_;  ba0 = ba0_;
//         g   = g_;    dt  = dt_;
//     }

//     virtual bool read(std::istream& is) override { return false; }
//     virtual bool write(std::ostream& os) const override { return false; }
// };

#endif