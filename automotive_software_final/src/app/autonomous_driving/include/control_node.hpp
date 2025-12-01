
#ifndef __CONTROL_NODE_HPP__
#define __CONTROL_NODE_HPP__
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

// Interface Header (ROS 독립적)
#include "interface_lane.hpp"
#include "interface_vehicle.hpp" 

// Bridge Header
#include "ros2_bridge_vehicle.hpp"
#include "ros2_bridge_lane.hpp"
#include "ros2_bridge_mission.hpp"

#include <std_msgs/msg/float32.hpp>

// Parameter Header
#include "autonomous_driving_config.hpp"

//class 멤버함수이기에 .cpp에서 ControlNode::LateralControl 로 정의
class ControlNode : public rclcpp::Node {
    public:
        explicit ControlNode(const std::string& node_name, const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
        virtual ~ControlNode();

        void ProcessParams();
        void Run();

    private:
        //----------------------------------------------------//
        // Functions
        void Init(const rclcpp::Time &current_time);  // ← 이 줄 추가

        // Callback functions
        inline void CallbackManualInput(const ad_msgs::msg::VehicleCommand::SharedPtr msg) {
            std::lock_guard<std::mutex> lock(mutex_manual_input_);
            if (cfg_.use_manual_inputs == true) {
                i_manual_input_ = ros2_bridge::GetVehicleCommand(*msg);
                b_is_manual_input_ = true;
            }
        }
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
        inline void CallbackMission(const ad_msgs::msg::Mission::SharedPtr msg) {
            std::lock_guard<std::mutex> lock(mutex_mission_);
            i_mission_ = ros2_bridge::GetMission(*msg);
            b_is_mission_ = true;
        }
        //[다훈 수정] reference_speed_ callback 추가
        inline void CallbackReferenceSpeed(const std_msgs::msg::Float32::SharedPtr msg) {            
            std::lock_guard<std::mutex> lock(mutex_reference_speed_);
            i_reference_speed_ = msg->data;
            b_is_reference_speed_ = true;
        }
        //[다훈 수정] driving_way callback 추가
        inline void CallbackDrivingWay(const ad_msgs::msg::PolyfitLaneData::SharedPtr msg) {            
            std::lock_guard<std::mutex> lock(mutex_driving_way_real_);
            i_driving_way_real_ = ros2_bridge::GetPolyfitLaneData(*msg);
            b_is_driving_way_real_ = true;
        }
        //=====================================================
        // TODO: Add more functions
        //=====================================================
        /** - algorithm::LateralControl()
        * @brief Calculate steering angle using pure pursuit
        * @param vehicle_state Vehicle state (interface)
        * @param driving_way_real Driving way (interface)
        * @return Steering angle
        */
        double LateralControl(const interface::VehicleState &vehicle_state, const interface::PolyfitLane &driving_way_real, const AutonomousDrivingConfig &cfg);

        /** -algorithm::LongitudinalControl()
         * @brief Calculate acceleration and brake using PID control
         * @param vehicle_state Vehicle state (interface)
         * @param reference_speed Reference speed (double)
         * @return Pair of acceleration and brake (std::pair<double, double>)

        */
        std::pair<double, double> LongitudinalControl(const interface::VehicleState &vehicle_state, const double &reference_speed, const AutonomousDrivingConfig &cfg);


        //------------------------------------------------------//
        // Variable

        //Subscriber
        rclcpp::Subscription<ad_msgs::msg::VehicleCommand>::SharedPtr       s_manual_input_;
        rclcpp::Subscription<ad_msgs::msg::VehicleState>::SharedPtr s_vehicle_state_;
        rclcpp::Subscription<ad_msgs::msg::LanePointData>::SharedPtr s_lane_points_;
        rclcpp::Subscription<ad_msgs::msg::Mission>::SharedPtr s_mission_;
        //[다훈 수정] reference_speed_ 구독 추가
        rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr s_reference_speed_;
        //[다훈 수정] lateral control을 위해 driving-way 구독 추가
        rclcpp::Subscription<ad_msgs::msg::PolyfitLaneData>::SharedPtr s_driving_way_real_;

        //===============================================
        // Input
        //===============================================
        interface::VehicleCommand i_manual_input_;
        interface::VehicleState i_vehicle_state_;
        interface::Lane i_lane_points_;
        interface::Mission i_mission_;
        //[다훈 수정] driving_way 추가
        interface::PolyfitLane i_driving_way_real_;

        //[다훈 수정] reference_speed_ 추가
        double i_reference_speed_ = 0.0;
        
        //===============================================
        // Mutex
        //===============================================
        std::mutex mutex_manual_input_;
        std::mutex mutex_vehicle_state_;
        std::mutex mutex_lane_points_;
        std::mutex mutex_mission_;
        //[다훈 수정] reference_speed_ mutex 추가
        std::mutex mutex_reference_speed_;
        std::mutex mutex_driving_way_real_;

        // Publisher (멤버 선언)
        //(1) vehicle command publish (control에서 사용)
        rclcpp::Publisher<ad_msgs::msg::VehicleCommand>::SharedPtr p_vehicle_command_;

        // Timer
        rclcpp::TimerBase::SharedPtr t_run_node_;

        // Util and Configuration
        AutonomousDrivingConfig cfg_;

        // Flag
        bool b_is_manual_input_ = false;
        bool b_is_simulator_on_ = false;
        bool b_is_lane_points_ = false;
        bool b_is_mission_ = false;
        bool b_is_reference_speed_ = false;
        bool b_is_driving_way_real_ = false;

};

#endif // __CONTROL_NODE_HPP_