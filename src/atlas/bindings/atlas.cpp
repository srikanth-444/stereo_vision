#include <pybind11/pybind11.h>


namespace py = pybind11;

// forward declarations
void bind_frame(py::module_ &m);
void bind_landmark(py::module_ &m);
void bind_map(py::module_ &m);


PYBIND11_MODULE(Atlas, m) {
    bind_frame(m);
    bind_landmark(m);
    bind_map(m);
}