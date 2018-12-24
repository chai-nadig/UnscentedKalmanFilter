#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
   * End DO NOT MODIFY section for measurement noise values
   */

  /**
   * Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
   is_initialized_ = false;
   n_x_ = 5;
   n_aug_ = 7;
   lambda_ = 3 - n_aug_;

   Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
   weights_ = VectorXd(2 * n_aug_ +1);

   P_ = MatrixXd::Identity(n_x_, n_x_);
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * Make sure you switch between lidar and radar measurements.
   */
   if (!is_initialized_) {
     if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
       // Convert radar from polar to cartesian coordinates
       //         and initialize state.
         float rho = meas_package.raw_measurements_[0];
         float phi = meas_package.raw_measurements_[1];

         float px = rho * cos(phi);
         float py = rho * sin(phi);

         x_ = VectorXd(4);
         x_ << px,
         py,
         0,
         0;
     } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        // Initialize state.
        x_ = VectorXd(4);
        x_ << meas_package.raw_measurements_[0],
        meas_package.raw_measurements_[1],
        0,
        0;
     }
      time_us_ = meas_package.timestamp_;
      // done initializing, no need to predict or update
      is_initialized_ = true;
      return;
   }

   double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;

   Prediction(delta_t);

   if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
     UpdateRadar(meas_package);
   } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
     UpdateLidar(meas_package);
   }

   time_us_ = meas_package.timestamp_;

}

void UKF::Prediction(double delta_t) {
  /**
   * Estimate the object's location.
   * Modify the state vector, x_. Predict sigma points, the state,
   * and the state covariance matrix.
   */
   VectorXd x_aug_ = VectorXd(n_aug_);
   MatrixXd P_aug_ = MatrixXd(n_aug_, n_aug_);
   MatrixXd Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

   x_aug_.head(n_x_) = x_;
   x_aug_(5) = 0;
   x_aug_(6) = 0;

   MatrixXd Q_ = MatrixXd(2, 2);
   Q_ << std_a_ * std_a_, 0, 0, std_yawdd_ * std_yawdd_;

   P_aug_.setZero();
   P_aug_.block(0, 0, n_x_, n_x_) = P_;
   P_aug_.block(n_x_, n_x_, 2, 2) = Q_;

   // create square root matrix
   MatrixXd A_ = P_aug_.llt().matrixL();

    // create augmented sigma points
    Xsig_aug_.col(0) = x_aug_;
    Xsig_aug_.block(0, 1, n_aug_, n_aug_) = (A_ * sqrt(lambda_ + n_aug_)).colwise() + x_aug_;
    Xsig_aug_.block(0, 1 + n_aug_, n_aug_, n_aug_) = (-(A_ * sqrt(lambda_ + n_aug_))).colwise() + x_aug_;

    // Predict Sigma points
    Xsig_pred_.setZero();
    for(int i=0; i < 2 * n_aug_ + 1; i ++) {
      VectorXd sigma_point_k = Xsig_aug_.col(i);

      double half_dt_sq = 0.5 * delta_t * delta_t;

      double v  = sigma_point_k[2];
      double psi = sigma_point_k[3];
      double psi_dot = sigma_point_k[4];
      double nu_a = sigma_point_k[5];
      double nu_psi_dot_dot = sigma_point_k[6];

      VectorXd noise_k = VectorXd(n_x_);
      noise_k << half_dt_sq * cos(psi) * nu_a,
      half_dt_sq * sin(psi) * nu_a,
      delta_t * nu_a,
      half_dt_sq * nu_psi_dot_dot,
      delta_t * nu_psi_dot_dot;

      VectorXd f = VectorXd(n_x_);
      if (fabs(psi_dot) < 0.0001) {
          f << v * cos(psi) * delta_t,
          v * sin(psi) * delta_t,
          0,
          0,
          0;
      } else {
          double v_psi_dot = v / psi_dot;
          f << v_psi_dot * (sin(psi + psi_dot * delta_t) - sin(psi)),
          v_psi_dot * (-cos(psi + psi_dot * delta_t) + cos(psi)),
          0,
          psi_dot * delta_t,
          0;
      }

      VectorXd xk = sigma_point_k.head(n_x_);

      VectorXd sigma_point_k1 = xk  + f + noise_k;

      Xsig_pred_.block(0, i, n_x_, 1) = sigma_point_k1;
  }

  // Predict state and state covariance matrix
  x_.setZero();
  P_.setZero();
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
      if (i == 0) {
          weights_[i] = lambda_ / (lambda_ + n_aug_);
      } else {
          weights_[i] = 0.5 / (lambda_ + n_aug_);
      }
      x_ += weights_[i] * Xsig_pred_.col(i);
  }
  for (int i = 0; i < 2 * n_aug_ +1 ; i++) {
      VectorXd diff = Xsig_pred_.col(i) - x_;
      while (diff[3] > M_PI) diff[3] -= 2*M_PI;
      while (diff[3] < -M_PI) diff[3] += 2*M_PI;

      P_ += weights_[i] * diff * diff.transpose();
  }

}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
}
