#pragma once
#include "imu_interface.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>

class ImuCSVSource : public IMUInterface {
public:
    explicit ImuCSVSource(const std::string& csv_path);

    Eigen::Matrix<double, 7, 1>    read(double timestamp)                           override;
    Eigen::Matrix<double,7,Eigen::Dynamic>        read_batch(double t0, double t1) override;
    bool                       is_available() const             override;
    void                       reset()                          override;
    std::optional<double>      get_frequency() const            override;

private:
    std::vector<std::vector<std::string>>   rows_;
    std::unordered_map<std::string, int>    col_index_;
    size_t                                  index_ = 0;

    void                     load(const std::string& csv_path);
    Eigen::Matrix<double, 7, 1> parse(size_t index);
    Eigen::Matrix<double, 7, 1> parse_row(const std::vector<std::string>& row) const;
    double                   col(const std::vector<std::string>& row,
                                 const std::string& name) const;
};