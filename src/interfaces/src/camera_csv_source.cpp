#include "camera_csv_source.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;

// ── helpers ──────────────────────────────────────────────────────────────────

static std::string trim(const std::string& s) {
    std::string out = s;
    if (out.size() >= 3 &&
        (unsigned char)out[0] == 0xEF &&
        (unsigned char)out[1] == 0xBB &&
        (unsigned char)out[2] == 0xBF)
        out = out.substr(3);
    auto l = out.find_first_not_of(" \t\r\n");
    auto r = out.find_last_not_of(" \t\r\n");
    return (l == std::string::npos) ? "" : out.substr(l, r - l + 1);
}

static std::vector<std::string> split_line(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string f;
    while (std::getline(ss, f, ','))
        fields.push_back(trim(f));
    return fields;
}

// ── lifecycle ─────────────────────────────────────────────────────────────────

CameraCSVSource::CameraCSVSource(const std::string& csv_path, int /*queue_size*/)
{
    // queue_size param kept for API compat but capacity is compile-time
    load(csv_path);
    thread_ = std::thread(&CameraCSVSource::producer, this);
}

CameraCSVSource::~CameraCSVSource() {
    stop();
}

void CameraCSVSource::stop() {
    stopped_.store(true, std::memory_order_relaxed);
    if (thread_.joinable())
        thread_.join();
}

// ── load ──────────────────────────────────────────────────────────────────────

void CameraCSVSource::load(const std::string& csv_path) {
    fs::path p(csv_path);
    data_dir_ = (p.parent_path() / "data").string();

    std::cout << "[CameraCSVSource] dir: " << p.parent_path()
              << "  file: " << p.filename() << "\n";

    std::ifstream f(csv_path);
    if (!f.is_open())
        throw std::runtime_error("Failed to open CSV: " + csv_path);

    std::string line;
    if (!std::getline(f, line))
        throw std::runtime_error("CSV is empty: " + csv_path);

    auto headers = split_line(line);
    for (int i = 0; i < static_cast<int>(headers.size()); ++i)
        col_index_[headers[i]] = i;

    while (std::getline(f, line))
        if (!line.empty())
            rows_.push_back(split_line(line));
}

// ── producer ──────────────────────────────────────────────────────────────────

void CameraCSVSource::producer() {
    while (!stopped_.load(std::memory_order_relaxed)
           && index_.load(std::memory_order_relaxed) < rows_.size())
    {
        const auto& row = rows_[index_.load(std::memory_order_relaxed)];
        double      ts  = std::stod(col(row, "#timestamp [ns]"));
        std::string fp  = (fs::path(data_dir_) / col(row, "filename")).string();

        cv::Mat frame = cv::imread(fp, cv::IMREAD_GRAYSCALE);
        if (frame.empty()) {
            std::cerr << "[CameraCSVSource] failed to read: " << fp << "\n";
            reached_end_.store(true, std::memory_order_relaxed);
            index_.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        // spin until there's space — no mutex, no syscall
        while (!queue_.push({ frame, ts })) {
            if (stopped_.load(std::memory_order_relaxed)) return;
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }

        index_.fetch_add(1, std::memory_order_relaxed);
    }

    reached_end_.store(true, std::memory_order_relaxed);
}

// ── consumer ──────────────────────────────────────────────────────────────────

std::pair<cv::Mat, double> CameraCSVSource::read_frame() {
    std::pair<cv::Mat, double> f;

    // spin until a frame arrives — typically sub-microsecond with full queue
    while (!queue_.pop(f)) {
        if (stopped_.load(std::memory_order_relaxed)
            || reached_end_.load(std::memory_order_relaxed))
        {
            // drain any last frames before declaring done
            if (queue_.pop(f)) return f;
            return { cv::Mat{}, -1.0 };
        }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }

    return f;
}

// ── misc ──────────────────────────────────────────────────────────────────────

bool CameraCSVSource::is_available() {
    return !reached_end_.load(std::memory_order_relaxed)
        || !queue_.empty();   // still frames left to drain
}

std::string CameraCSVSource::col(const std::vector<std::string>& row,
                                  const std::string& name) const {
    auto it = col_index_.find(name);
    if (it == col_index_.end())
        throw std::out_of_range("Missing column: " + name);
    return row.at(it->second);
}