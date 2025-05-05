#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

// Sobel-Filter via Eigen-Matrix
Eigen::MatrixXd sobel(Eigen::MatrixXd gray_img, Eigen::MatrixXd filter) {
    int h = gray_img.rows(), w = gray_img.cols();
    // Ausgabe ohne Randbehandlung: h-2 × w-2
    Eigen::MatrixXd filtered_img(h-2, w-2);

    for (int i = 1; i < h-1; ++i) {
        for (int j = 1; j < w-1; ++j) {
            double sum = 0.0;
            // 3×3-Faltung
            for (int u = -1; u <= 1; ++u) {
                for (int v = -1; v <= 1; ++v) {
                    sum += filter(u+1, v+1) * gray_img(i+u, j+v);
                }
            }
            filtered_img(i-1, j-1) = sum;
        }
    }
    return filtered_img;
}

PYBIND11_MODULE(sobel_demo, m) {
    m.doc() = "sobel operator using numpy!";
    m.def("sobel", &sobel,
          py::arg("gray_img"), py::arg("filter"),
          "Compute Sobel derivative (Eigen version)");
}