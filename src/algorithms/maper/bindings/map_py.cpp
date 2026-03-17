#include "map.h"
#include "frame.h"
#include "landmark.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;
void bind_map(py::module_ &m) {
    py::class_<Map>(m,"Map")
        .def(py::init<>())
        .def("createLandmarks",&Map::createLandmarks)
        .def("removeBadLandmarks", &Map::removeBadLandmarks)
        .def("mergeLandmarks", &Map::mergeLandmarks)
        .def("setKeyframe",[](Map &m, std::shared_ptr<Frame> frame) 
                                {
                                m.setKeyframe(frame);
                                })
        .def("getLastKeyFrame",&Map::getLastKeyFrame)
        .def("getAgedFrame",&Map::getAgedFrame)
        .def("getClosestKeyFrames",&Map::getClosestKeyFrames)
        .def("getLocalMap",&Map::getLocalMap);
}