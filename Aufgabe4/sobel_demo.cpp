#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <algorithm>

namespace py = pybind11;

Eigen::MatrixXd sobel(Eigen::MatrixXd gray_img, Eigen::MatrixXd filter) {
    Eigen::MatrixXd filtered_img(gray_img.rows() - 2, gray_img.cols() - 2);

    // Anwenden des Sobel-Filters
    for (int i = 1; i < gray_img.rows() - 1; ++i) {
        for (int j = 1; j < gray_img.cols() - 1; ++j) {
            double sum = 0.0;
            for (int k = -1; k <= 1; ++k) {
                for (int l = -1; l <= 1; ++l) {
                    sum += gray_img(i + k, j + l) * filter(k + 1, l + 1);
                }
            }
            filtered_img(i - 1, j - 1) = sum;
        }
    }

    // Skalierung des Ergebnisses auf den Bereich [0, 255]
    double min_val = filtered_img.minCoeff();
    double max_val = filtered_img.maxCoeff();
    if (max_val - min_val > 0) {
        filtered_img = (filtered_img.array() - min_val).matrix() / (max_val - min_val) * 255.0;
    } else {
        filtered_img.setZero();
    }

    // Werte auf den Bereich [0, 255] beschränken
    for (int i = 0; i < filtered_img.rows(); ++i) {
        for (int j = 0; j < filtered_img.cols(); ++j) {
            filtered_img(i, j) = std::min(std::max(filtered_img(i, j), 0.0), 255.0);
        }
    }

    return filtered_img;
}

// Pybind11-Moduldefinition
PYBIND11_MODULE(sobel_demo, m) {
    m.doc() = "Sobel-Operator mit Pybind11";
    m.def("sobel", &sobel, "Berechnet die erste Ableitung eines Bildes mit einem Sobel-Filter",
          py::arg("gray_img"), py::arg("filter"));
}