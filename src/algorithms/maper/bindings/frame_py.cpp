#include "frame.h"
#include "landmark.h"
#include "featureExtractor.h"
#include <pybind11/pybind11.h>
#include <opencv2/core/core.hpp>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

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

void bind_frame(py::module_ &m) {

    py::class_<Frame,std::shared_ptr<Frame>>(m, "Frame")
        .def(py::init([](
        int id, py::array_t<uint8_t> img, int timeStamp,py::array_t<float> K, py::array_t<float> Tcw,FeatureExtractor* extractor
        ) 
    {   
        std::cout<<"entering lambda"<<std::endl;
        
        Eigen::Map<Eigen::Matrix<float,3,3,Eigen::ColMajor>> intrinsic(K.mutable_data());
        Eigen::Map<Eigen::Matrix<float,4,4,Eigen::ColMajor>> extrinsic(Tcw.mutable_data());
        
        
        // 2. Convert NumPy array to cv::Mat
        py::buffer_info buf = img.request();
        cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC1, (unsigned char*)buf.ptr);
        
        // 3. Optional: If your C++ Frame constructor expects to OWN the pixels, 
        // you should .clone() it, otherwise mat just points to the Python memory.
        cv::Mat mat_copy=mat.clone();

        return std::make_shared<Frame>(id, mat_copy, timeStamp, intrinsic, extrinsic, extractor);
    })) 
        .def_readwrite("id", &Frame::id)
        .def_property_readonly("image", [](Frame &f){ return mat_to_numpy(f.image); })
        .def_readwrite("timeStamp", &Frame::timeStamp)
        .def_readwrite("intrinsic", &Frame::intrinsic)
        .def_readwrite("extrinsic", &Frame::extrinsic)
        .def_readwrite("cameraCenter", &Frame::cameraCenter)
        .def("getNotAssociatedPoints", &Frame::getNotAssociatedPoints)
        .def("getNotAssociatedDescriptors", &Frame::getNotAssociatedDescriptors)
        .def("getLandmarks", &Frame::getLandmarks)
        .def("getTrackedPoints",&Frame::getTrackedPoints)
        .def("getCameraCenter",&Frame::getCameraCenter)
        .def("addKeyPointsToGrid",&Frame::addKeyPointsToGrid)
        .def("updateCovisibility",&Frame::updateCovisibility)
        .def("extractFeatures",&Frame::extractFeatures)
        .def("setCameraWorldPose",&Frame::setCameraWorldPose)
        .def("projectionMatch", [](Frame &f, std::vector<Landmark*> landmarks, std::vector<cv::Point3f>& mObjectPoints, std::vector<cv::Point2f>& mImagePoints)
                                {
                                    f.projectionMatch(landmarks,mObjectPoints,mImagePoints);
                                })
        .def("localMatch", [](Frame &f, std::vector<Landmark*> landmarks)
                            {
                                f.projectionMatch(landmarks);
                            });
}