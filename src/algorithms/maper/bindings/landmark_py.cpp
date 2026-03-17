#include "landmark.h"
#include "frame.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;
void bind_landmark(py::module_ &m) {
    py::class_<Landmark,std::unique_ptr<Landmark>>(m, "Landmark")
        .def(py::init([](int id,py::array_t<float> point, Frame* f, int featureId){
            Eigen::Map<Eigen::Vector3f> point3D(point.mutable_data());
            return std::make_unique<Landmark>(id,point3D,f,featureId);
        }))
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