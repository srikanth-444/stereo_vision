#include "map.h"
#include "frame.h"
#include "landmark.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;
void bind_map(py::module_ &m) {
    py::class_<Map,std::shared_ptr<Map>>(m,"Map")
        .def(py::init<>())
        .def_readwrite("landmarks",&Map::landmarks)
        .def_readwrite("keyFrames",&Map::keyFrames)
        .def("createLandmarks",&Map::createLandmarks)
        .def("removeBadLandmarks", &Map::removeBadLandmarks)
        .def("mergeLandmarks", &Map::mergeLandmarks)
        .def("setKeyframe",&Map::setKeyframe)
        .def("getLastKeyFrame",&Map::getLastKeyFrame)
        .def("getAgedFrame",&Map::getAgedFrame)
        .def("getClosestKeyFrames",&Map::getClosestKeyFrames)
        .def("getLocalMap",&Map::getLocalMap)
        .def("getLengthKeyFrame",&Map::getLengthKeyFrame);
}