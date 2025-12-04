
#ifndef __PERCEPTION_NODE_HPP__
#define __PERCEPTION_NODE_HPP__
#pragma once

// STD Header
#include <memory>
#include <mutex>
#include <utility>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <eigen3/Eigen/Dense> 

// Interface Header
#include "interface_lane.hpp"
#include "interface_vehicle.hpp" 

// Bridge Header
#include "ros2_bridge_vehicle.hpp"
#include "ros2_bridge_lane.hpp"
#include "ros2_bridge_mission.hpp"

// Parameter Header
#include "autonomous_driving_config.hpp"

class PerceptionNode : public rclcpp::Node {
    public:
        explicit PerceptionNode(const std::string& node_name, const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
        virtual ~PerceptionNode();

        void ProcessParams();
        void Run();

    private:
        //===============================================
        // Functions 
        //===============================================
        void Init(const rclcpp::Time &current_time); 

        inline void CallbackVehicleState(const ad_msgs::msg::VehicleState::SharedPtr msg) {            
            std::lock_guard<std::mutex> lock(mutex_vehicle_state_);
            i_vehicle_state_ = ros2_bridge::GetVehicleState(*msg);
            b_is_simulator_on_ = true;
        }
        inline void CallbackLanePoints(const ad_msgs::msg::LanePointData::SharedPtr msg) {            
            std::lock_guard<std::mutex> lock(mutex_lane_points_);
            i_lane_points_ = ros2_bridge::GetLanePoints(*msg);
            b_is_lane_points_ = true;
        }

        /**
         * @brief 차선 포인트 데이터를 3차 다항식으로 피팅하여 차선을 검출하는 함수
         * @param lane_points 차선 포인트 배열
         * @return 검출된 차선들의 다항식 계수 (a0, a1, a2, a3)
         */
        interface::PolyfitLanes FindLanes(const interface::Lane& lane_points);

        /**
         * @brief 검출된 차선들의 중앙선을 계산하여 주행 경로를 생성하는 함수
         * @param vehicle_state 현재 차량 상태
         * @param lanes 검출된 차선들
         * @return 주행 경로의 다항식 계수
         */
        interface::PolyfitLane FindDrivingWay(const interface::VehicleState &vehicle_state, const interface::PolyfitLanes& lanes);

        /**
         * @brief 1차원 K-means 클러스터링을 수행하는 함수
         * @param values 클러스터링할 Y값들의 벡터
         * @param k_max 최대 클러스터 개수
         * @param max_iter 최대 반복 횟수
         * @return 클러스터 결과 (중심값, 소속 값들) 쌍의 벡터
         */
        std::vector<std::pair<double, std::vector<double>>> Kmeans1D(
            const std::vector<double>& values, int k_max, int max_iter);

        /**
         * @brief 3차 다항식 피팅을 수행하는 함수 (LSM)
         * @param pts 피팅할 2D 점들
         * @return 다항식 계수 (a0, a1, a2, a3)
         */
        Eigen::Vector4d SolvePolynomial(const std::vector<Eigen::Vector2d>& pts);

        /**
         * @brief RANSAC을 사용하여 3차 다항식 피팅을 수행하는 함수
         * @param pts 피팅할 2D 점들
         * @param min_points_for_fit 피팅에 필요한 최소 점 개수
         * @return 다항식 계수 (a0, a1, a2, a3)
         */
        Eigen::Vector4d FitWithRansac(const std::vector<Eigen::Vector2d>& pts, int min_points_for_fit);
        
        //===============================================
        // Variables
        //===============================================
        interface::VehicleCommand i_manual_input_;
        interface::VehicleState i_vehicle_state_;
        interface::Lane i_lane_points_;
        
        int ransac_max_iterations;
        double ransac_inlier_threshold;
        double ransac_min_inlier_ratio;

        // 이전 프레임 차선 결과 저장 (포인트 부족 시 유지용)
        static constexpr int kMinPointsForFit = 4;
        static constexpr double kEmaAlpha = 0.4;  // EMA 블렌딩 비율 (0.0~1.0, 높을수록 새 값 반영)
        std::vector<Eigen::Vector4d> prev_lane_coeffs_{4, Eigen::Vector4d::Zero()};
        std::vector<bool> prev_lane_valid_ = std::vector<bool>(4, false);

        std::mutex mutex_manual_input_;
        std::mutex mutex_vehicle_state_;
        std::mutex mutex_lane_points_;
        
        rclcpp::TimerBase::SharedPtr t_run_node_;

        AutonomousDrivingConfig cfg_;

        bool b_is_manual_input_ = false;
        bool b_is_simulator_on_ = false;
        bool b_is_lane_points_ = false;

        //===============================================
        // Subscriber
        //===============================================
        rclcpp::Subscription<ad_msgs::msg::VehicleCommand>::SharedPtr s_manual_input_;
        rclcpp::Subscription<ad_msgs::msg::VehicleState>::SharedPtr s_vehicle_state_;
        rclcpp::Subscription<ad_msgs::msg::LanePointData>::SharedPtr s_lane_points_;
        
        //===============================================
        // Publisher
        //===============================================
        rclcpp::Publisher<ad_msgs::msg::PolyfitLaneDataArray>::SharedPtr p_poly_lanes_;
        rclcpp::Publisher<ad_msgs::msg::PolyfitLaneData>::SharedPtr p_driving_way_;

    };

#endif // __PERCEPTION_NODE_HPP__
