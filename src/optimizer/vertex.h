#pragma once

#include <g2o/core/base_vertex.h>
#include <Eigen/Core>

class VertexVelocity : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexVelocity() {}
    virtual void setToOriginImpl() override {
        setEstimate(Eigen::Vector3d::Zero());
    }
    virtual void oplusImpl(const double* update_) override {
        Eigen::Map<const Eigen::Vector3d> update(update_);
        setEstimate(estimate() + update);
    }
    virtual bool read(std::istream& is) override { return false; }
    virtual bool write(std::ostream& os) const override { return false; }
};

class VertexGyroBias : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexGyroBias() {}
    virtual void setToOriginImpl() override {
        setEstimate(Eigen::Vector3d::Zero());
    }
    virtual void oplusImpl(const double* update_) override {
        Eigen::Map<const Eigen::Vector3d> update(update_);
        setEstimate(estimate() + update);
    }
    virtual bool read(std::istream& is) override { return false; }
    virtual bool write(std::ostream& os) const override { return false; }
};

class VertexAccBias : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexAccBias() {}
    virtual void setToOriginImpl() override {
        setEstimate(Eigen::Vector3d::Zero());
    }
    virtual void oplusImpl(const double* update_) override {
        Eigen::Map<const Eigen::Vector3d> update(update_);
        setEstimate(estimate() + update);
    }
    virtual bool read(std::istream& is) override { return false; }
    virtual bool write(std::ostream& os) const override { return false; }
};
