#include <mlpack.hpp>
#include "utils.h"
#include "metrics.h"


std::tuple<double,double,double> LogisticReg(arma::mat &X_train, arma::Row<size_t> &y_train, arma::mat &X_test, arma::Row<size_t> &y_test) {
    mlpack::regression::LogisticRegression<> loreg(X_train,y_train,0.0001);
    arma::Row<size_t> pred_lr;
    loreg.Classify(X_test,pred_lr);
    auto m = ComputeMtcs(y_test,pred_lr);
    double a = m.precision;
    double b = m.recall;
    double c = m.f1;
    return {a,b,c};
}

std::tuple<double,double,double> LinSVM(arma::mat &X_train, arma::Row<size_t> &y_train, arma::mat &X_test, arma::Row<size_t> &y_test) {
    mlpack::svm::LinearSVM<> lsvm(X_train, y_train, 2);
    arma::Row<size_t> pred_svm;
    lsvm.Classify(X_test, pred_svm);
    auto m = ComputeMtcs(y_test, pred_svm);
    double a = m.precision;
    double b = m.recall;
    double c = m.f1;
    return {a,b,c};
}

std::tuple<double,double,double> DesTree(arma::mat &X_train, arma::Row<size_t> &y_train, arma::mat &X_test, arma::Row<size_t> &y_test) {
    mlpack::tree::DecisionTree<> trees(X_train, y_train, 2);
    arma::Row<size_t> pred_tree;
    trees.Classify(X_test, pred_tree);
    auto m = ComputeMtcs(y_test, pred_tree);
    double a = m.precision;
    double b = m.recall;
    double c = m.f1;
    return {a,b,c};
}

int main() {
    arma::mat X;
    arma::Row<size_t> y;
    LoadDataset("data.csv", X, y);

    //Randomizing here because each coloumn here is a sample
    arma::uvec indices = arma::randperm(X.n_cols);

    //We are splitting the data set given into 80 percent of training and 20 percent of testing
    size_t split = 0.8 * X.n_cols;

    //Training Data set:
    arma::mat X_train = X.cols(indices.head(split));
    arma::Row<size_t> y_train = y.cols(indices.head(split));

    //Testing Data set:
    arma::mat X_test = X.cols(indices.tail(X.n_cols - split));
    arma::Row<size_t> y_test = y.cols(indices.tail(X.n_cols - split));

    //Now we take the output for various methods:
    arma::mat MetricOutput(3,3);
    auto [a,b,c] = LogisticReg(X_train,y_train,X_test,y_test);
    auto [p,q,r] = LinSVM(X_train,y_train,X_test,y_test);
    auto [t,u,v] = DesTree(X_train,y_train,X_test,y_test);
    MetricOutput.row(0) = arma::rowvec({a,b,c});
    MetricOutput.row(1) = arma::rowvec({p,q,r});
    MetricOutput.row(2) = arma::rowvec({t,u,v});

    //Now We will output the given matrix which consistis of the data from the 3 different methods taken here:
    SaveCSV("outputs/metrics.csv", MetricOutput);
    return 0;
}
