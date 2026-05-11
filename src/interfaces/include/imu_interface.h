#pragma once
#include <array>
#include <optional>
#include <vector>
#include <Eigen/Dense>


class IMUInterface {
public:
    virtual ~IMUInterface() = default;

    virtual Eigen::Matrix<double, 7, 1>   read(double timestamp)                          = 0;
    virtual Eigen::Matrix<double,7,Eigen::Dynamic>        read_batch(double t0, double t1) = 0;
    virtual bool                          is_available() const             = 0;
    virtual void                          reset()                          = 0;
    virtual std::optional<double>         get_frequency() const            = 0;
};