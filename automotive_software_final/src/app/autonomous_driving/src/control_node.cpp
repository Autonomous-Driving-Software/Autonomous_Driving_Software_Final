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
    //[다훈 수정] lateral control을 위해 driving_way_real 가져오기
    if (b_is_driving_way_real_ == false) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Wait for Driving Way ...");
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

    // [다훈 수정] lateral control을 위해 driving_way_real 가져오기(지역변수로 복사)
    interface::PolyfitLane driving_way_real; {
        std::lock_guard<std::mutex> lock(mutex_driving_way_real_);
        driving_way_real = i_driving_way_real_;
    }
    
    //===================================================
    // Algorithm
    //===================================================
    // (1) lateral control
    double steering_angle = ControlNode::LateralControl(vehicle_state, driving_way_real, cfg_);
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
// LateralControl 함수 구현 
//===================================================
double ControlNode::LateralControl(const interface::VehicleState &vehicle_state, const interface::PolyfitLane &driving_way_real, const AutonomousDrivingConfig &cfg) {
    /*
    *@brief Calculate steering using Pure Pursuit algorithm
    * inputs: vehicle_state, driving_way_real, cfg
    * output: steering angle (radian??)
    * Purpose: Implement Pure Pursuit Control to calculate the steering angle based on the vehicle state and driving way
    */

    ///////////////////TODO///////////////////
    //Initialize Inputs
    double l_xd; // look-ahead distance [m]
    double g_x, g_y; // look-ahead point coordinates [m]
    double l_d; // distance between vehicle and look-ahead point [m]
    //Initialize Outputs
    double steering_angle = 0.0;

    // Step 0: Set look-ahead distance
    l_xd = cfg.param_pp_kd;

    //[다훈 수정]Step 1: Get look-ahead point using look-ahead distance
    // (g_x g_y) = (x, ax^3 + bx^2 + cx + d)|x=l_xd
    // driving_way_real의 계수 사용
    // step 1-1: g_x는 l_xd로 고정
    g_x = l_xd;
    // step 1-2: g_y는 다항식에 대입하여 계산
    g_y = driving_way_real.a3 * pow(g_x, 3) + driving_way_real.a2 * pow(g_x, 2) + driving_way_real.a1 * g_x + driving_way_real.a0;

    //[다훈 수정]Step 2: The distance between vehicle position and look-ahead point
    //l_d = sqrt(g_x² + g_y²)
    //e_ld = g_y
    l_d = sqrt(pow(g_x, 2) + pow(g_y, 2));
    double e_ld = g_y; // lateral error

    //[다훈 수정]Step 3: Calculate steering angle using Pure Pursuit formula
    // steering_angle = atan2(2 * L * e_ld / l_d^2)
    //1/R = 2 * e_ld / l_d^2
    // wheel_base 멤버 변수 사용(헤더파일에 정의되어 있음)
    steering_angle = atan2(2.0 * cfg.param_wheel_base * e_ld, (l_d * l_d));

    /////////////////////////////////////////////////
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
    double accel_command = 0.5;
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