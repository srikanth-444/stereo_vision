#pragma once
#include <unordered_map>
#include <string>
#include <Eigen/Dense>
#include <memory>

// Forward-declare your interface type (adapt to your actual interface)


class IMUInterface;
class IMU {
public:

    
    IMU(const double g_noise_density, const double g_random_walk, const double a_noise_density, const double a_random_walk, const double rate, std::shared_ptr<IMUInterface> interface);
    Eigen::Matrix<double, 7, 1> getData(double timestamp);
    Eigen::Matrix<double, 7, Eigen::Dynamic> getDataRange(double t0, double t1);

    double get_dt() const;
    bool is_available() const;

    Eigen::Matrix<double, 6, 6> getContinuousNoiseCov() const;
    Eigen::Matrix<double, 6, 6> getDiscreteNoiseCov(double dt) const;
    Eigen::Matrix<double, 6, 6> getBiasNoiseCov() const;
    Eigen::Matrix<double, 4, 1> getNoiseParams() const;

private:
    std::shared_ptr<IMUInterface> interface;
    double g_noise_density;  
    double g_random_walk;   
    double a_noise_density;   
    double a_random_walk;   
    double rate;            
};