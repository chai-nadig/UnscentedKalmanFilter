#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using std::vector;
using std::cout;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * Calculate the RMSE here.
   */
   VectorXd rmse(4);
    rmse << 0,0,0,0;

    if (estimations.size() == 0) {
        cout << "size of estimations is zero";
        return rmse;
    }

    if (estimations.size() != ground_truth.size()) {
        cout << "size of estimations is not equal to size of ground truth";
        return rmse;
    }

    // accumulate squared residuals
    for (int i=0; i < estimations.size(); ++i) {
        VectorXd diff = estimations.at(i)  - ground_truth.at(i);
        rmse = rmse.array() + diff.array() * diff.array() ;
    }

    // calculate the mean
    rmse /= estimations.size();

    // calculate the squared root
    rmse = rmse.array().sqrt();

    // return the result
    return rmse;
}
