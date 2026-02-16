#pragma once
#include <armadillo>
#include <fstream>

void LoadDataset(const std::string &path, arma::mat &x, arma::Row<size_t> &y) {
    arma::mat data;
    data.load(path,arma::csv_ascii);
    x = data.cols(0,data.n_cols-2).t();
    y = arma::conv_to<arma::Row<size_t>>::from(data.col(data.n_cols-1).t());
}

void SaveCSV(const std::string &path, const arma::mat &data) {
    data.save(path,arma::csv_ascii);
}
