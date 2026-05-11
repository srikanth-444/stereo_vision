#include "imu_csv_source.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

static std::vector<std::string> split_line(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string f;
    while (std::getline(ss, f, ','))
        fields.push_back(f);
    return fields;
}

ImuCSVSource::ImuCSVSource(const std::string& csv_path) {
    load(csv_path);
    std::cout << "[ImuCSVSource] loaded " << rows_.size()
              << " rows from " << csv_path << "\n";
}

void ImuCSVSource::load(const std::string& csv_path) {
    std::ifstream f(csv_path);
    if (!f.is_open())
        throw std::runtime_error("Failed to open IMU CSV: " + csv_path);

    std::string line;
    if (!std::getline(f, line))
        throw std::runtime_error("IMU CSV is empty: " + csv_path);

    auto headers = split_line(line);
    for (int i = 0; i < static_cast<int>(headers.size()); ++i)
        col_index_[headers[i]] = i;

    while (std::getline(f, line))
        if (!line.empty())
            rows_.push_back(split_line(line));
}

Eigen::Matrix<double, 7, 1>  ImuCSVSource::read( double timestamp) {
    if (index_ >= rows_.size()) return Eigen::Matrix<double, 7, 1>::Zero();
    return parse(index_);
}

Eigen::Matrix<double,7,Eigen::Dynamic> ImuCSVSource::read_batch(double t0, double t1) {
    Eigen::Matrix<double,7,Eigen::Dynamic> batch(7, 0);
    for (const auto& row : rows_) {
        try {
            double t = col(row, "#timestamp [ns]");
            if (t0 <= t && t <= t1) {
                auto s = parse_row(row);
                if (!s.hasNaN())   // check for invalid values instead
                        batch.conservativeResize(7, batch.cols() + 1);
                        batch.col(batch.cols() - 1) = s;
            }
        } catch (...) { continue; }
    }
    return batch;
}

bool ImuCSVSource::is_available() const {
    return index_ < rows_.size();
}

void ImuCSVSource::reset() {
    index_ = 0;
}

std::optional<double> ImuCSVSource::get_frequency() const {
    if (rows_.size() < 2) return std::nullopt;
    try {
        double t0 = col(rows_[0], "#timestamp [ns]");
        double t1 = col(rows_[1], "#timestamp [ns]");
        double dt = t1 - t0;
        return dt > 0.0 ? std::optional<double>(1e9 / dt) : std::nullopt;
    } catch (...) { return std::nullopt; }
}


Eigen::Matrix<double, 7, 1>  ImuCSVSource::parse(size_t index) {
    auto sample = parse_row(rows_[index]);
    ++index_;   // always advance
    if (sample.hasNaN())   // check for invalid values instead
        std::cerr << "[ImuCSVSource] malformed row at index " << index << "\n";
    return sample;
}

Eigen::Matrix<double, 7, 1> ImuCSVSource::parse_row(const std::vector<std::string>& row) const {
    try {
        Eigen::Matrix<double, 7, 1> s;
        s(0) = col(row, "#timestamp [ns]");
        s(1) = col(row, "a_RS_S_x [m s^-2]");
        s(2) = col(row, "a_RS_S_y [m s^-2]");
        s(3) = col(row, "a_RS_S_z [m s^-2]");
        s(4) = col(row, "w_RS_S_x [rad s^-1]");
        s(5) = col(row, "w_RS_S_y [rad s^-1]");
        s(6) = col(row, "w_RS_S_z [rad s^-1]");
        return s;
    } catch (const std::exception& e) {
        std::cerr << "[ImuCSVSource] parse error: " << e.what() << "\n";
        return Eigen::Matrix<double, 7, 1>::Zero();
    }
}

double ImuCSVSource::col(const std::vector<std::string>& row,
                         const std::string& name) const {
    auto it = col_index_.find(name);
    if (it == col_index_.end())
        throw std::out_of_range("Missing column: " + name);
    return std::stod(row.at(it->second));
}