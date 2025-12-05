
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
        
        //===============================================
        // Variables
        //===============================================
        interface::VehicleState i_vehicle_state_;
        interface::Lane i_lane_points_;

        // 이전 프레임 차선 결과 저장 (포인트 부족 시 유지용)
        static constexpr int kMinPointsForFit = 4;
        static constexpr double kEmaAlpha = 0.4;  // EMA 블렌딩 비율 (0.0~1.0, 높을수록 새 값 반영)
        static constexpr int kLaneCount = 2;      // 검출 차선 수 (좌/우)
        static constexpr int kMaxFramesWithoutUpdate = 5;  // 업데이트 없이 유지할 최대 프레임
        
        std::vector<Eigen::Vector4d> prev_lane_coeffs_{kLaneCount, Eigen::Vector4d::Zero()};
        std::vector<bool> prev_lane_valid_{kLaneCount, false};
        std::vector<int> prev_frames_without_update_{kLaneCount, 0};
        
        // 주행 경로 이전 프레임 저장
        Eigen::Vector4d prev_driving_way_coeffs_ = Eigen::Vector4d::Zero();
        bool prev_driving_way_valid_ = false;
        int prev_driving_way_frames_ = 0;

        std::mutex mutex_vehicle_state_;
        std::mutex mutex_lane_points_;
        
        rclcpp::TimerBase::SharedPtr t_run_node_;

        AutonomousDrivingConfig cfg_;

        bool b_is_simulator_on_ = false;
        bool b_is_lane_points_ = false;

        //===============================================
        // Subscriber
        //===============================================
        rclcpp::Subscription<ad_msgs::msg::VehicleState>::SharedPtr s_vehicle_state_;
        rclcpp::Subscription<ad_msgs::msg::LanePointData>::SharedPtr s_lane_points_;
        
        //===============================================
        // Publisher
        //===============================================
        rclcpp::Publisher<ad_msgs::msg::PolyfitLaneDataArray>::SharedPtr p_poly_lanes_;
        rclcpp::Publisher<ad_msgs::msg::PolyfitLaneData>::SharedPtr p_driving_way_;

    };

#endif // __PERCEPTION_NODE_HPP__
