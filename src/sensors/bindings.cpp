#include "camera.h"
#include "imu.h"
#include <pybind11/pybind11.h>
#include "camera_interface.h"
#include "imu_interface.h"
#include "featureExtractor.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <iostream>
#include "frame.h"


namespace py = pybind11;
py::array mat_to_numpy(const cv::Mat& mat)
{
    if (mat.empty())
        throw std::runtime_error("Empty cv::Mat");

    // Only support single-channel or 3-channel CV_8U for simplicity
    size_t rows = mat.rows;
    size_t cols = mat.cols;
    size_t channels = mat.channels();
    
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    
    if (channels == 1) {
        shape = {rows, cols};
        strides = {mat.step, mat.elemSize()};
    } else {
        shape = {rows, cols, channels};
        strides = {mat.step, mat.elemSize() * channels, mat.elemSize()};
    }

    return py::array(py::buffer_info(
        mat.data,
        mat.elemSize(),
        py::format_descriptor<unsigned char>::format(),
        shape.size(),
        shape,
        strides
    ));
}


PYBIND11_MODULE(Sensors, m) {
    py::class_<Camera, std::shared_ptr<Camera>>(m,"Camera")
        .def(py::init<const Eigen::Matrix3f&, const Eigen::Vector4f&, const Eigen::Matrix4f&, std::shared_ptr<CameraInterface>, std::shared_ptr<FeatureExtractor>, int, int>())
        .def("get_intrinsic", &Camera::getIntrinsic)
        .def("get_extrinsic", &Camera::getExtrinsic)
        .def("get_width", &Camera::getWidth)
        .def("get_height", &Camera::getHeight)
        .def("get_frame",&Camera::getFrame,py::call_guard<py::gil_scoped_release>())
        .def("undistort_frame", &Camera::undistortFrame)
        .def("project", &Camera::project)
        .def("get_feature_extractor", &Camera::getFeatureExtractor);

    py::class_<IMU, std::shared_ptr<IMU>>(m,"IMU")
        .def(py::init<const double&, const double&, const double&, const double&, const double&, std::shared_ptr<IMUInterface>>())
        .def("get_data", &IMU::getData)
        .def("get_data_range", &IMU::getDataRange)
        .def("get_dt", &IMU::get_dt)
        .def("is_available", &IMU::is_available)
        .def("get_continuous_noise_cov", &IMU::getContinuousNoiseCov)
        .def("get_discrete_noise_cov", &IMU::getDiscreteNoiseCov)
        .def("get_bias_noise_cov", &IMU::getBiasNoiseCov)
        .def("get_noise_params", &IMU::getNoiseParams);
};