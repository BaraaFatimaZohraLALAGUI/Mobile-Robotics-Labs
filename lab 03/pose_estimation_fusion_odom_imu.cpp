#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

class PoseEstimationFusion : public rclcpp::Node {
public:
    PoseEstimationFusion() : Node("pose_estimation_fusion_odom_imu") {
        // Initialize state vector: [x, y, theta]
        state_ = VectorXd(3);
        state_ << 0, 0, 0;

        // Initialize covariance matrix (P)
        P_ = MatrixXd(3, 3);
        P_.setIdentity();
        P_ *= 0.1; // Set initial uncertainty

        // Initialize process noise covariance (Q) and measurement noise covariance (R)
        Q_ = MatrixXd(3, 3);
        Q_ << 0.1, 0, 0,
              0, 0.1, 0,
              0, 0, 0.01; // Process noise for state

        R_ = MatrixXd(1, 1);
        R_ << 0.1; // Measurement noise for IMU yaw angle

        // Subscribe to /encoder/robot_vel topic
        robot_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/encoder/robot_vel", 10, std::bind(&PoseEstimationFusion::odometry_callback, this, std::placeholders::_1));

        // Subscribe to /imu topic
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu", 10, std::bind(&PoseEstimationFusion::imu_callback, this, std::placeholders::_1));

        // Publisher for Kalman Filter estimated pose
        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/kalman_filtered_pose2", 10);
    }

private:
    void odometry_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        // Extract linear velocity (vt) and angular velocity (wt)
        double vt = msg->linear.x;
        double wt = msg->angular.z;
        double dt = 0.1; // Time step (fixed for simplicity)

        // Prediction step using odometry data
        MatrixXd F(3, 3); // Jacobian of motion model
        F << 1, 0, -vt * sin(state_[2]) * dt,
             0, 1, vt * cos(state_[2]) * dt,
             0, 0, 1;

        MatrixXd B(3, 2); // Control input model
        B << dt * cos(state_[2]), 0,
             dt * sin(state_[2]), 0,
             0, dt;

        VectorXd u(2); // Control input vector
        u << vt, wt;

        // State prediction
        state_ = state_ + B * u;

        // Covariance prediction
        P_ = F * P_ * F.transpose() + Q_;
    }

    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        // Extract IMU measurement (yaw angle from the IMU)
        double zt = msg->orientation.z;

        // Measurement model for IMU data (only orientation matters)
        MatrixXd H(1, 3); // Measurement matrix
        H << 0, 0, 1;

        // Innovation (measurement residual)
        VectorXd y(1); 
        y << zt - state_[2];

        // Measurement covariance
        MatrixXd S = H * P_ * H.transpose() + R_;

        // Kalman Gain
        MatrixXd K = P_ * H.transpose() * S.inverse();

        // State update
        state_ = state_ + K * y;

        // Covariance update
        P_ = (MatrixXd::Identity(3, 3) - K * H) * P_;

        // Publish the fused pose
        publish_fused_pose();
    }

    void publish_fused_pose() {
        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header.stamp = this->get_clock()->now();
        pose_msg.header.frame_id = "map"; // Assume map frame
        pose_msg.pose.position.x = state_[0];
        pose_msg.pose.position.y = state_[1];
        pose_msg.pose.orientation.z = sin(state_[2] / 2);
        pose_msg.pose.orientation.w = cos(state_[2] / 2);

        pose_pub_->publish(pose_msg);
    }

    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr robot_vel_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;

    VectorXd state_; // State: [x, y, theta]
    MatrixXd P_; // Covariance matrix
    MatrixXd Q_; // Process noise covariance
    MatrixXd R_; // Measurement noise covariance
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PoseEstimationFusion>());
    rclcpp::shutdown();
    return 0;
}
