#include <pybind11/pybind11.h>
#include "camera_csv_source.h"
#include "imu_csv_source.h"
#include "camera_interface.h"
#include "imu_interface.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "featureExtractor.h"

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


PYBIND11_MODULE(Interface, m) {
    py::class_<CameraInterface, std::shared_ptr<CameraInterface>>(m,"CameraInterface");
    py::class_<IMUInterface, std::shared_ptr<IMUInterface>>(m,"IMUInterface");
    py::class_<CameraCSVSource,CameraInterface,std::shared_ptr<CameraCSVSource>>(m,"CameraCSVSource")
        .def(py::init<const std::string&, int>())
        .def("read_frame", [](CameraCSVSource &f){
            std::pair<cv::Mat, double> image_ts_pair;
            {
            py::gil_scoped_release release; // release GIL during blocking read
            image_ts_pair = f.read_frame();
            } 
            return py::make_tuple(mat_to_numpy(image_ts_pair.first), image_ts_pair.second);})
        .def("is_available", &CameraCSVSource::is_available)
        .def("stop", &CameraCSVSource::stop);
    py::class_<ImuCSVSource, IMUInterface, std::shared_ptr<ImuCSVSource>>(m,"ImuCSVSource")
        .def(py::init<const std::string&>())
        .def("read", &ImuCSVSource::read)
        .def("read_batch", &ImuCSVSource::read_batch)
        .def("is_available", &ImuCSVSource::is_available)
        .def("reset", &ImuCSVSource::reset)
        .def("get_frequency", &ImuCSVSource::get_frequency);
   
}