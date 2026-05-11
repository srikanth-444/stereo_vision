#include <pybind11/pybind11.h>
#include "visualize.h"
#include "atlas.h"
#include "frame.h"

namespace py = pybind11;

PYBIND11_MODULE(Visualize, m){
    py::class_<Visualize, std::shared_ptr<Visualize>>(m, "Visualize")
        .def(py::init<Atlas*>())
        .def("update",                    &Visualize::update,py::call_guard<py::gil_scoped_release>())
        .def("visualize_pipeline",        &Visualize::visualize_pipeline,py::call_guard<py::gil_scoped_release>())
        .def("visualize_as_point_cloud",  &Visualize::visualize_as_point_cloud,py::call_guard<py::gil_scoped_release>())
        .def("show_full_map",             &Visualize::show_full_map,py::call_guard<py::gil_scoped_release>());
}