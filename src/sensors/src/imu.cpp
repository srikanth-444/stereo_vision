#include "imu.h"
#include "imu_interface.h"
#include <stdexcept>

IMU::IMU(const double g_noise_density, const double g_random_walk, const double a_noise_density, const double a_random_walk, const double rate, std::shared_ptr<IMUInterface> interface)
    : interface(interface),
      g_noise_density(g_noise_density),
      g_random_walk(g_random_walk),
      a_noise_density(a_noise_density),
      a_random_walk(a_random_walk),
      rate(rate)
{
    if (!interface)
        throw std::invalid_argument("IMUInterface pointer must not be null");
}

Eigen::Matrix<double, 7, 1>IMU::getData(double timestamp) {
    return interface->read(timestamp);
}

Eigen::Matrix<double, 7, Eigen::Dynamic> IMU::getDataRange(double t0, double t1) {
    return interface->read_batch(t0, t1);
}

double IMU::get_dt() const {
    return 1.0 / rate;
}

bool IMU::is_available() const {
    return interface->is_available();
}

Eigen::Matrix<double, 6, 6> IMU::getContinuousNoiseCov() const {
    double sa2 = a_noise_density * a_noise_density;
    double sg2 = g_noise_density * g_noise_density;

    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Zero();
    cov(0,0) = sa2; cov(1,1) = sa2; cov(2,2) = sa2;
    cov(3,3) = sg2; cov(4,4) = sg2; cov(5,5) = sg2;
    return cov;
}

Eigen::Matrix<double, 6, 6> IMU::getDiscreteNoiseCov(double dt) const {
    return getContinuousNoiseCov() / dt;
}

Eigen::Matrix<double, 6, 6> IMU::getBiasNoiseCov() const {
    double sba2 = a_random_walk * a_random_walk;
    double sbg2 = g_random_walk * g_random_walk;

    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Zero();
    cov(0,0) = sba2; cov(1,1) = sba2; cov(2,2) = sba2;
    cov(3,3) = sbg2; cov(4,4) = sbg2; cov(5,5) = sbg2;
    return cov;
}

Eigen::Matrix<double, 4, 1> IMU::getNoiseParams() const {
    Eigen::Matrix<double, 4, 1> params;
    params << a_noise_density, g_noise_density, a_random_walk, g_random_walk;
    return params;
}