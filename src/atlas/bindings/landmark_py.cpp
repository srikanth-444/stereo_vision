#include "landmark.h"
#include "frame.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <cstdint>
namespace py = pybind11;
py::array mat_to_numpy(const cv::Mat& mat);

void bind_landmark(py::module_ &m) {
    py::class_<Landmark,std::shared_ptr<Landmark>>(m, "Landmark")
        .def(py::init<int,Eigen::Vector3f, std::shared_ptr<Frame>, int>())
        .def_readwrite("id", &Landmark::id)
        .def_readwrite("point3D", &Landmark::point3D)
        .def_readwrite("normal", &Landmark::normal)
        .def_property_readonly("descriptor",[](Landmark &f){return mat_to_numpy(f.descriptor);})
        .def_readwrite("refDepth", &Landmark::refDepth)
        .def_readwrite("nVisible", &Landmark::nVisible)
        .def_readwrite("nTracked", &Landmark::nTracked)
        .def("setNormal",&Landmark::setNormal)
        .def("setReferenceDepth",&Landmark::setReferenceDepth)
        .def("isVisible",&Landmark::isVisible)
        .def("addObservation",&Landmark::addObservation)
        .def("erase",&Landmark::erase)
        .def("updateDescriptor",&Landmark::updateDescriptor)
        .def("increaseVisible",&Landmark::increaseVisible)
        .def("increaseTracked",&Landmark::increaseTracked)
        .def("getFoundingRatio",&Landmark::getFoundingRatio)
        .def("setBadflag",&Landmark::setBadFlag);
}