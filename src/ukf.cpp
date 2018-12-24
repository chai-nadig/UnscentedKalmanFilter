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

   if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
     UpdateRadar(meas_package);
     time_us_ = meas_package.timestamp_;
   } else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
     UpdateLidar(meas_package);
     time_us_ = meas_package.timestamp_;
   }

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
   * Use lidar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
   int n_z_ = 2;

   // mean predicted measurement
   VectorXd z_pred_ = VectorXd(n_z_);
   z_pred_.setZero();

   // measurement covariance matrix S
   MatrixXd S = MatrixXd(n_z_, n_z_);

   // create matrix for sigma points in measurement space
   MatrixXd Zsig_ = MatrixXd(n_z_, 2 * n_aug_ + 1);

   // transform sigma points into measurement space
   // calculate mean predicted measurement
   // calculate innovation covariance matrix S

   for (int i=0; i < 2 * n_aug_ + 1; i++) {
     VectorXd xk1 = Xsig_pred_.col(i);

     double px = xk1[0];
     double py = xk1[1];

     VectorXd Zk1 = VectorXd(n_z_);
     Zk1 << px, py;
     Zsig_.col(i) = Zk1;

     z_pred_ += weights_[i] * Zk1;
   }

   S.setZero();
   for (int i=0; i < 2 * n_aug_ + 1; i++) {
       VectorXd diff = (Zsig_.col(i) - z_pred_);
       S += weights_[i] * diff * diff.transpose();
   }

   MatrixXd R = MatrixXd(n_z_, n_z_);
   R << std_laspx_ * std_laspx_, 0, 0, std_laspy_ * std_laspy_;

   S += R;

   // create matrix for cross correlation Tc
   MatrixXd Tc = MatrixXd(n_x_, n_z_);

   // calculate cross correlation matrix
   Tc.setZero();
   for (int i =0; i < 2 * n_aug_ + 1; i++) {
       Tc += weights_[i] * (Xsig_pred_.col(i) - x_) * (Zsig_.col(i) - z_pred_).transpose();
   }

   // calculate Kalman gain K;
   MatrixXd K = Tc * S.inverse();

   // vector for incoming radar measurement
   VectorXd z_ = meas_package.raw_measurements_;

   // update state mean and covariance matrix
   VectorXd z_diff = z_ - z_pred_;

   x_ = x_ + K * z_diff;
   P_ = P_ - K * S * K.transpose();

}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * Use radar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

   // set measurement dimension, radar can measure r, phi, and r_dot
   int n_z_ = 3;

    // mean predicted measurement
    VectorXd z_pred_ = VectorXd(n_z_);
    z_pred_.setZero();

    // measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z_, n_z_);

    // create matrix for sigma points in measurement space
    MatrixXd Zsig_ = MatrixXd(n_z_, 2 * n_aug_ + 1);

    // transform sigma points into measurement space
    // calculate mean predicted measurement
    // calculate innovation covariance matrix S

    for (int i=0; i < 2 * n_aug_ + 1; i++) {

      VectorXd xk1 = Xsig_pred_.col(i);

      double px = xk1[0];
      double py = xk1[1];
      double v = xk1[2];
      double psi = xk1[3];

      double rho = sqrt(px*px + py*py);
      double phi = atan2(py, px);
      double rho_dot = (px*v*cos(psi) + py*v*sin(psi))/rho;

      VectorXd Zk1 = VectorXd(n_z_);
      Zk1 << rho, phi, rho_dot;
      Zsig_.col(i) = Zk1;

      z_pred_ += weights_[i] * Zk1;
    }

    S.setZero();
    for (int i=0; i < 2 * n_aug_ + 1; i++) {
        VectorXd diff = (Zsig_.col(i) - z_pred_);
        S += weights_[i] * diff * diff.transpose();
    }
    MatrixXd R = MatrixXd(n_z_, n_z_);
    R << std_radr_ * std_radr_, 0, 0,
    0, std_radphi_ * std_radphi_, 0,
    0, 0, std_radrd_ * std_radrd_;

    S += R;

    // create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z_);

    // calculate cross correlation matrix
    Tc.setZero();
    for (int i =0; i < 2 * n_aug_ + 1; i++) {
        Tc += weights_[i] * (Xsig_pred_.col(i) - x_) * (Zsig_.col(i) - z_pred_).transpose();
    }

    // calculate Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    // vector for incoming radar measurement
    VectorXd z_ = meas_package.raw_measurements_;

    // update state mean and covariance matrix
    VectorXd z_diff = z_ - z_pred_;
    while (z_diff(1)> M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1)<-M_PI) z_diff(1) += 2. * M_PI;

    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();
}
