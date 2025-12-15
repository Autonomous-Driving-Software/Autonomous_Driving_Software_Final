/*
* control_node.cpp
*/
#include "autonomous_driving_config.hpp"
#include "control_node.hpp"

using namespace std;

ControlNode::ControlNode(const std::string &node_name, const rclcpp::NodeOptions &options)
    : Node(node_name, options) {

    RCLCPP_WARN(this->get_logger(), "Initialize node...");

    // QoS init
    auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(10));

    //===============parameters===============
    //declare parameters(파라미터 등록+초기값 설정)
    this->declare_parameter("autonomous_driving/ns", "");
    this->declare_parameter("autonomous_driving/loop_rate_hz", 100.0);
    this->declare_parameter("autonomous_driving/use_manual_inputs", false);    

    ProcessParams();

    RCLCPP_INFO(this->get_logger(), "vehicle_namespace: %s", cfg_.vehicle_namespace.c_str());
    RCLCPP_INFO(this->get_logger(), "loop_rate_hz: %f", cfg_.loop_rate_hz);
    RCLCPP_INFO(this->get_logger(), "use_manual_inputs: %d", cfg_.use_manual_inputs);
    
    //=================================================
    //get parameters(파라미터 값 가져오기)
    //=================================================
    //(1) Control parameters
    this->get_parameter("autonomous_driving/pure_pursuit_kd", cfg_.param_pp_kd);
    this->get_parameter("autonomous_driving/pure_pursuit_kv", cfg_.param_pp_kv);
    this->get_parameter("autonomous_driving/pure_pursuit_kc", cfg_.param_pp_kc);
    this->get_parameter("autonomous_driving/pid_kp", cfg_.param_pid_kp);
    this->get_parameter("autonomous_driving/pid_ki", cfg_.param_pid_ki);
    this->get_parameter("autonomous_driving/pid_kd", cfg_.param_pid_kd);
    this->get_parameter("autonomous_driving/brake_ratio", cfg_.param_brake_ratio);

    //(2) Longitudinal control parameters
    this->get_parameter("autonomous_driving/speed_error_integral", cfg_.speed_error_integral);
    this->get_parameter("autonomous_driving/speed_error_prev", cfg_.speed_error_prev);

    //(3) Vehicle
    this->get_parameter("autonomous_driving/wheel_base", cfg_.param_wheel_base);
    this->get_parameter("autonomous_driving/max_lateral_accel", cfg_.param_max_lateral_accel);

    /* 노드를 ROS 네트워크에 연결하는 부분 
    공통적으로 하는 일 create_subscription<메시지타입>(토픽이름, QoS, callback)*/
    //=================================================
    //Subscriber init 
    //=================================================
    //(1)s_manual_input_
    s_manual_input_ = this->create_subscription<ad_msgs::msg::VehicleCommand>(
        "/manual_input", qos_profile, std::bind(&ControlNode::CallbackManualInput, this, std::placeholders::_1));

    //(2) s_vehicle_state_
    s_vehicle_state_ = 
    this->create_subscription<ad_msgs::msg::VehicleState>(
        "vehicle_state", qos_profile, std::bind(&ControlNode::CallbackVehicleState, this, std::placeholders::_1));

    //[다훈 수정0] /limit_speed_ subscriber 추가 
    //(3) s_limit_speed_
    //s_limit_speed_ = this->create_subscription<std_msgs::msg::Float32>(
    //    "/limit_speed", qos_profile, std::bind(&ControlNode::CallbackLimitSpeed, this, std::placeholders::_1));

    //(4) s_lane_points_
    s_lane_points_ = 
    this->create_subscription<ad_msgs::msg::LanePointData>(
        "lane_points", qos_profile, std::bind(&ControlNode::CallbackLanePoints, this, std::placeholders::_1));
        
    //(5) s_mission_
    s_mission_ = this->create_subscription<ad_msgs::msg::Mission>(
        "mission", qos_profile, std::bind(&ControlNode::CallbackMission, this, std::placeholders::_1));

    //[다훈 수정1] /reference_speed_ subscriber 추가
    s_reference_speed_ = this->create_subscription<std_msgs::msg::Float32>("reference_speed", qos_profile, 
        std::bind(&ControlNode::CallbackReferenceSpeed, this, std::placeholders::_1));

    //[다훈 수정] lateral control을 위해 driving-way subscriber 추가 (planning node에서 퍼블리시하는거)
    s_driving_way_real_ = this->create_subscription<ad_msgs::msg::PolyfitLaneData>(
        "driving_way_real", qos_profile, std::bind(&ControlNode::CallbackDrivingWay, this, std::placeholders::_1));
    s_driving_way_points_ = this->create_subscription<ad_msgs::msg::LanePointData>(
        "driving_way_points", qos_profile, std::bind(&ControlNode::CallbackPlannedPathPoints, this, std::placeholders::_1));

    //=================================================
    //Publisher init
    //=================================================
    p_vehicle_command_ = this->create_publisher<ad_msgs::msg::VehicleCommand>(
        "vehicle_command", qos_profile);

    //=================================================
    //Initialization
    //=================================================
    Init(this->now());

    //=================================================
    //Timer init
    //=================================================
    t_run_node_ = this->create_wall_timer(
        std::chrono::milliseconds((int64_t)(1000 / cfg_.loop_rate_hz)),
        [this]() { this->Run(); }); 

}
ControlNode::~ControlNode() {}

void ControlNode::Init(const rclcpp::Time &current_time) {}

void ControlNode::ProcessParams() {
    this->get_parameter("autonomous_driving/ns", cfg_.vehicle_namespace);
    this->get_parameter("autonomous_driving/loop_rate_hz", cfg_.loop_rate_hz);
    this->get_parameter("autonomous_driving/use_manual_inputs", cfg_.use_manual_inputs);
}

void ControlNode::Run() {
    auto current_time = this->now();
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Running Control Node...");
    ProcessParams();

    //===================================================
    // 아직 필요한 input 토픽 안들어왔으면 알고리즘 실행하지 않고 기다림
    // Get subscribe variables
    //===================================================
    if (cfg_.use_manual_inputs == true) {
        if (b_is_manual_input_ == false) {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Wait for Manual Input ...");
            return;
        }
    }

    if (b_is_simulator_on_ == false) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Wait for Vehicle State ...");
        return;
    }

    if (b_is_lane_points_ == false) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Wait for Lane Points ...");
        return;
    }

    if (b_is_mission_ == false) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Wait for Mission ...");
        return;
    }
    if (b_is_reference_speed_ == false) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Wait for Reference Speed ...");
        return;
    }
    if (b_is_driving_way_points_ == false) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Wait for Planned Path Points ...");
        return;
    }

    //===================================================
    // Get subscribe variables
    // [일종의 input데이터 수집 단계 (멤버 변수 -> 지역변수로 복사 (mutex로 보호))]
    //===================================================
    interface::VehicleCommand manual_input; {
        if (cfg_.use_manual_inputs == true) {
            std::lock_guard<std::mutex> lock(mutex_manual_input_);
            manual_input = i_manual_input_;
        }
    }
    interface::VehicleState vehicle_state; {
        std::lock_guard<std::mutex> lock(mutex_vehicle_state_);
        vehicle_state = i_vehicle_state_;
    }
    //[다훈 수정1] i_limit_speed_ local변수로 복사해서 run에서 사용 
    //double limit_speed; {
    //    std::lock_guard<std::mutex> lock(mutex_limit_speed_);
    //    limit_speed = i_limit_speed_;
    //}

    interface::Lane lane_points; {
        std::lock_guard<std::mutex> lock(mutex_lane_points_);
        lane_points = i_lane_points_;
    }

    interface::Mission mission; {
        std::lock_guard<std::mutex> lock(mutex_mission_);
        mission = i_mission_;
    }

    double reference_speed; {
        std::lock_guard<std::mutex> lock(mutex_reference_speed_);
        reference_speed = i_reference_speed_;
    }

    interface::Lane driving_way_points; {
        std::lock_guard<std::mutex> lock(mutex_driving_way_points_);
        driving_way_points = i_driving_way_points_;
    }
    
    //===================================================
    // Algorithm
    //===================================================
    // (1) lateral control
    double steering_angle = ControlNode::LateralControl(vehicle_state, driving_way_points, cfg_);
    // (2) output variables: longitudinal control
    interface::VehicleCommand vehicle_command;

    vehicle_command.steering = steering_angle;
    vehicle_command.accel = 0.0;
    vehicle_command.brake = 0.0;

    // Initialize pair of command variabless
    std::pair<double, double> accel_brake_command;

    // longitudinal control
    accel_brake_command = ControlNode::LongitudinalControl(vehicle_state, reference_speed, cfg_);

    // [다훈 수정11.27]longitudinal control 결과를 vehicle_command에 반영 (이거 빠져서 속도가 계속 들어간 듯)
    vehicle_command.accel = accel_brake_command.first;
    vehicle_command.brake = accel_brake_command.second;

    ///////////////////////////////////////////////////////

    if (cfg_.use_manual_inputs == true) {
        vehicle_command = manual_input;
    }

    //===================================================
    // Publish output
    //===================================================
    // (1) Publish vehicle command
    p_vehicle_command_->publish(ros2_bridge::UpdateVehicleCommand(vehicle_command));
}

//===================================================
// LateralControl 함수 구현 (Pure Pursuit with path points)
//===================================================
double ControlNode::LateralControl(const interface::VehicleState &vehicle_state, const interface::Lane &path_points, const AutonomousDrivingConfig &cfg) {
    /*
    *@brief Calculate steering using Pure Pursuit algorithm
    * inputs: vehicle_state, path_points (already local/body frame), cfg
    * output: steering angle (radian)
    */

    if (path_points.point.empty()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *get_clock(), 500, "[LateralControl] No path points, steering=0");
        return 0.0;
    }

    // Pure Pursuit의 look-ahead 거리
    const double l_xd = cfg.param_pp_kd + cfg.param_pp_kv * vehicle_state.velocity + cfg.param_pp_kc;
    //const double l_xd = cfg.param_pp_kd;

    // look-ahead 지점을 path 상에서 보간
    double target_x = path_points.point.front().x;
    double target_y = path_points.point.front().y;
    double accumulated = 0.0;
    bool found_target = false;

    for (size_t i = 1; i < path_points.point.size(); ++i) {
        double dx = path_points.point[i].x - path_points.point[i - 1].x;
        double dy = path_points.point[i].y - path_points.point[i - 1].y;
        double segment_len = std::hypot(dx, dy);

        accumulated += segment_len;

        if (accumulated >= l_xd && segment_len > 1e-3) {
            double ratio = (l_xd - (accumulated - segment_len)) / segment_len;
            if (ratio < 0.0) {
                ratio = 0.0;
            } else if (ratio > 1.0) {
                ratio = 1.0;
            }
            target_x = path_points.point[i - 1].x + ratio * dx;
            target_y = path_points.point[i - 1].y + ratio * dy;
            found_target = true;
            break;
        }
    }

    if (!found_target) {
        target_x = path_points.point.back().x;
        target_y = path_points.point.back().y;
    }

    double l_d = std::hypot(target_x, target_y);
    if (l_d < 1e-3) {
        return 0.0;
    }

    double e_ld = target_y; // 로컬 좌표계에서 y가 lateral error

    double steering_angle = std::atan2(2.0 * cfg.param_wheel_base * e_ld, (l_d * l_d));

    return steering_angle;
}

//===================================================
// LongitudinalControl 함수 구현 
//===================================================
std::pair<double, double> ControlNode::LongitudinalControl(const interface::VehicleState &vehicle_state, const double &reference_speed, const AutonomousDrivingConfig &cfg) {
    /**
     * @brief Calculate the acceleration and brake commands using PID control
     * inputs: vehicle_state, reference_speed
     * outputs: accel_command, brake_command
     * Purpose: Implement PID control to compute acceleration and brake commands to follow the reference speed
     */
    // Initialize Outputs
    double accel_command = 0.0;
    double brake_command = 0.0;

    ////////////////////// TODO //////////////////////
    // limit(여기서 limit은 velocity planning을 거쳐서 나온 reference_speed)을 보면서 PID값을 튜닝

    // First, Initialize private member variables in autonomous_driving.hpp
    // [Error] Calculate speed error, cumulative error, and derivative error
    double speed_error = reference_speed - vehicle_state.velocity;
    cfg_.speed_error_integral += speed_error * cfg.dt; // 누적 오차

    // [PID Control] Calculate acceleration, brake commands using PID formula
    // Parameters of PID is initialized in autonomous_driving.hpp: param_pid_kp_, param_pid_ki_, param_pid_kd_
    double u = (cfg.param_pid_kp * speed_error) + (cfg.param_pid_ki * cfg_.speed_error_integral) + (cfg.param_pid_kd * (speed_error - cfg_.speed_error_prev) / cfg.dt);

    cfg_.speed_error_prev = speed_error; // 이전 오차 저장
    // [Output] Set accel_command and brake_command values
    if (u > 0) {
        accel_command = u;
        brake_command = 0.0;
    } else {
        accel_command = 0.0;
        brake_command = -u * cfg.param_brake_ratio; // brake ratio 곱해서 제동력 조절
    }

    ///////////////////////////////////////////////////
    // pair로 접근하면 두 개를 다 접근할 수 있음 그래서 accel / brake 두 개를 first, second로 나눠서 접근 
    std::pair<double, double> accel_brake_command;
    accel_brake_command.first = accel_command;
    accel_brake_command.second = brake_command;

    return accel_brake_command;
}


int main(int argc, char **argv) {
    std::string node_name = "control_node";

    // Initialize node
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ControlNode>(node_name));
    rclcpp::shutdown();
    return 0;
}
