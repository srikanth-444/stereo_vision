#pragma once
#include "camera_interface.h"
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <unordered_map>
#include <boost/lockfree/spsc_queue.hpp>

class CameraCSVSource : public CameraInterface {
public:
    explicit CameraCSVSource(const std::string& csv_path,
                             int queue_size = 50);
    ~CameraCSVSource() override;

    std::pair<cv::Mat, double> read_frame()        override;
    bool                       is_available()        override;
    void                       stop()               override;

private:
    void        producer();
    void        load(const std::string& csv_path);
    std::string col(const std::vector<std::string>& row,
                    const std::string& name) const;

    std::string                           data_dir_;
    std::vector<std::vector<std::string>> rows_;
    std::unordered_map<std::string, int>  col_index_;
    std::atomic<size_t>                   index_{ 0 };

    // lock-free SPSC ring buffer — capacity must be a compile-time constant,
    // so we cap at 64; if you need runtime sizing, use boost::lockfree::queue
    // with a fixed_sized<true> policy instead.
    static constexpr std::size_t QUEUE_CAPACITY = 64;
    boost::lockfree::spsc_queue<std::pair<cv::Mat, double>, boost::lockfree::capacity<QUEUE_CAPACITY> > queue_;

    std::atomic<bool> stopped_   { false };
    std::atomic<bool> reached_end_{ false };
    std::thread       thread_;
};