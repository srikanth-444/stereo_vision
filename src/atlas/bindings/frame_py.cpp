#include "frame.h"
#include "landmark.h"
#include "featureExtractor.h"
#include <pybind11/pybind11.h>
#include <opencv2/core/core.hpp>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <iostream>
#include <cstdint>


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
        int id, py::array_t<uint8_t> img, int64_t timeStamp,Eigen::Matrix3f K, Eigen::Matrix4f Tcw,Eigen::Vector4f dist_coefficents,FeatureExtractor* extractor
        ) 
    {   
        py::buffer_info buf = img.request();
        cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC1, (unsigned char*)buf.ptr);
        cv::Mat mat_copy=mat.clone();
        return std::make_shared<Frame>(id, mat_copy, timeStamp, K, Tcw,dist_coefficents,extractor);
    })) 
        .def_readwrite("id", &Frame::id)
        .def_property_readonly("image", [](Frame &f){ return mat_to_numpy(f.image); })
        .def_readwrite("timeStamp", &Frame::timeStamp)
        .def_readwrite("intrinsic", &Frame::intrinsic)
        .def_readwrite("extrinsic", &Frame::extrinsic)
        .def_readwrite("cameraCenter", &Frame::cameraCenter)
        .def_readwrite("cameraNormal",&Frame::cameraNormal)
        .def_property_readonly("worldPose",[](Frame &f){
            Eigen::Vector4f v = f.q.coeffs();
            return py::make_tuple(v,f.t);})
        .def_readwrite("mergers", &Frame::mergers)
        .def_readwrite("nVisible",&Frame::nVisible)
        .def_readwrite("projectedPoints",&Frame::projectedPoints)
        .def("getNotAssociatedIndices",&Frame::getNotAssociatedIndices)
        .def("getNotAssociatedPoints", &Frame::getNotAssociatedPoints)
        .def("getNotAssociatedDescriptors",[](Frame &f){return mat_to_numpy(f.getNotAssociatedDescriptors());})
        .def("getLandmarks", &Frame::getLandmarks)
        .def("getTrackedPoints",&Frame::getTrackedPoints)
        .def("getCameraCenter",&Frame::getCameraCenter)
        .def("addKeyPointsToGrid",&Frame::addKeyPointsToGrid)
        .def("updateCovisibility",&Frame::updateCovisibility)
        .def("extractFeatures",&Frame::extractFeatures)
        .def("setCameraWorldPose",[](Frame &f, Eigen::Vector4f r, Eigen::Vector3f t){
            Eigen::Quaternionf q(r);
            return f.setCameraWorldPose(q,t);
        })
        .def("getVisibleLandmarks",&Frame::getVisibleLandmarks)
        .def("match",[](Frame &f, std::vector<std::shared_ptr<Landmark>> landmarks){
                std::vector<cv::Point3f> objectPoints;
                std::vector<cv::Point2f> imagePoints;

                f.match(landmarks, objectPoints, imagePoints);
                py::array_t obpoints=py::array_t<float>(
                { (int)objectPoints.size(), 3 },         
                { sizeof(cv::Point3f), sizeof(float) }, 
                (float*)objectPoints.data());
                py::array_t impoints=py::array_t<float>(
                { (int)imagePoints.size(), 2 },         
                { sizeof(cv::Point2f), sizeof(float) }, 
                (float*)imagePoints.data());
                return py::make_tuple(obpoints, impoints);
        })
        .def("projectionMatch", [](Frame &f, std::vector<std::shared_ptr<Landmark>> landmarks)
            {
                std::vector<cv::Point3f> objectPoints;
                std::vector<cv::Point2f> imagePoints;

                f.projectionMatch(landmarks, objectPoints, imagePoints);
                py::array_t obpoints=py::array_t<float>(
                { (int)objectPoints.size(), 3 },         
                { sizeof(cv::Point3f), sizeof(float) }, 
                (float*)objectPoints.data());
                py::array_t impoints=py::array_t<float>(
                { (int)imagePoints.size(), 2 },         
                { sizeof(cv::Point2f), sizeof(float) }, 
                (float*)imagePoints.data());
                return py::make_tuple(obpoints, impoints);
            })
        .def("localMatch", [](Frame &f, std::vector<std::shared_ptr<Landmark>>landmarks)
                            {
                                f.projectionMatch(landmarks);
                            });
}