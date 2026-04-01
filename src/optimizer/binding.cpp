#include <pybind11/pybind11.h>
#include "optimizer.h"

namespace py = pybind11;

PYBIND11_MODULE(Optimizer, m){
    py::class_<Optimizer,std::shared_ptr<Optimizer>>(m, "Optimizer")
    .def(py::init<bool>())
    .def("optimizePose",&Optimizer::optimizePose)
    .def("localBundleAdjustment",&Optimizer::localBundleAdjustment);
}