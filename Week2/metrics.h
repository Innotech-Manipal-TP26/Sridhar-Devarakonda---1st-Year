#pragma once
#include <armadillo>

struct Metrics {
    double precision;
    double recall;
    double f1;
};

Metrics ComputeMtcs(const arma::Row<size_t> &y_true, const arma::Row<size_t> &y_pred) {
    // tp: true positive
    // fp: false positive
    // fn: false negative
    double tp=0;
    double fp =0;
    double fn =0;
    auto n = y_true.n_elem;
    for (size_t i = 0; i < n; ++i) {
        auto a = y_true[i];
        auto b = y_pred[i];
        if (a==1&&b==1)
            tp++;
        if (a==0&&b==1)
            fn++;
        if (a==1&&b==0)
            fp++;
    }

    Metrics m;
    const double c = 1e-10;
    m.precision = tp/(tp+fp+c);
    m.recall = tp/(tp+fn+c);
    double a = m.precision;
    double b = m.recall;
    m.f1 = (2*a*b)/(a+b);
    return m;
}