/*
 * planning_node.cpp
 */
#include "autonomous_driving_config.hpp"
#include "planning_node.hpp"

using namespace std;

PlanningNode::PlanningNode(const std::string &node_name, const rclcpp::NodeOptions &options): Node(node_name, options) {
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

    //get parameters(파라미터 값 가져오기)
    // Vehicle
    //this->get_parameter("autonomous_driving/wheel_base", cfg_.param_wheel_base);
    //this->get_parameter("autonomous_driving/max_lateral_accel", cfg_.max_lateral_accel);

    /* 노드를 ROS 네트워크에 연결하는 부분 
    공통적으로 하는 일 create_subscription<메시지타입>(토픽이름, QoS, callback)*/
    //============Subscriber init===============
    //(1)s_manual_input_
    s_manual_input_ = this->create_subscription<ad_msgs::msg::VehicleCommand>(
        "/manual_input", qos_profile, std::bind(&PlanningNode::CallbackManualInput, this, std::placeholders::_1));

    //(2) s_vehicle_state_
    s_vehicle_state_ = 
    this->create_subscription<ad_msgs::msg::VehicleState>(
        "vehicle_state", qos_profile, std::bind(&
        PlanningNode::CallbackVehicleState, this, 
    std::placeholders::_1));

    //[다훈 수정0] /limit_speed_ subscriber 추가 
    //(3) s_limit_speed_
    //s_limit_speed_ = this->create_subscription<std_msgs::msg::Float32>(
    //    "/limit_speed", qos_profile, std::bind(&PlanningNode::CallbackLimitSpeed, this, std::placeholders::_1));

    //(4) s_lane_points_
    s_lane_points_ = 
    this->create_subscription<ad_msgs::msg::LanePointData>(
        "lane_points", qos_profile, std::bind(&PlanningNode::CallbackLanePoints, this, std::placeholders::_1));
        
    //(5) s_mission_
    s_mission_ = this->create_subscription<ad_msgs::msg::Mission>(
        "mission", qos_profile, std::bind(&PlanningNode::CallbackMission, this, std::placeholders::_1));

    //[11.28 다훈 수정] driving_way_raw subscriber 추가
    s_driving_way_raw_ = this->create_subscription<ad_msgs::msg::PolyfitLaneData>(
        "driving_way", qos_profile, std::bind(&PlanningNode::CallbackPolyfitLaneData, this, std::placeholders::_1));

    //======================================================
    //publisher init
    //======================================================
    p_vehicle_command_ = this->create_publisher<ad_msgs::msg::VehicleCommand>(
        "vehicle_command", qos_profile);
    p_driving_way_real_ = this->create_publisher<ad_msgs::msg::PolyfitLaneData>(
        "driving_way_real", qos_profile);

    //[다훈 추가]p_reference_speed_ (이게 맞나?)(lon에 넘기려고)
    p_reference_speed_ = this->create_publisher<std_msgs::msg::Float32>(
        "reference_speed", qos_profile);

    // Initialize
    Init(this->now());

    // Timer init
    t_run_node_ = this->create_wall_timer(
        std::chrono::milliseconds((int64_t)(1000 / cfg_.loop_rate_hz)),
        [this]() { this->Run(); }); 

}
PlanningNode::~PlanningNode() {}

void PlanningNode::Init(const rclcpp::Time &current_time) {
}

void PlanningNode::ProcessParams() {
    this->get_parameter("autonomous_driving/ns", cfg_.vehicle_namespace);
    this->get_parameter("autonomous_driving/loop_rate_hz", cfg_.loop_rate_hz);
    this->get_parameter("autonomous_driving/use_manual_inputs", cfg_.use_manual_inputs);
}

void PlanningNode::Run() {
    auto current_time = this->now();
    RCLCPP_INFO_THROTTLE(this->get_logger(), *get_clock(), 1000, "Running ..."); //로그는 최소 1000ms에 한번만 출력
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
    //[11.28 다훈 수정] driving_way_raw 추가
    if (b_is_driving_way_raw_ == false) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Wait for Driving Way Raw ...");
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

    interface::Lane lane_points; {
        std::lock_guard<std::mutex> lock(mutex_lane_points_);
        lane_points = i_lane_points_;
    }

    interface::Mission mission; {
        std::lock_guard<std::mutex> lock(mutex_mission_);
        mission = i_mission_;
    }

    interface::PolyfitLane driving_way_raw; {
    std::lock_guard<std::mutex> lock(mutex_driving_way_raw_);
    driving_way_raw = i_driving_way_raw_;
    }   

    //===================================================
    // Algorithm
    //===================================================
    //-> 이거는 perception에서 사용해야할거 같은데 나중에 옮기기 
    //(1)Add polyfit lane algorithm
    //interface::PolyfitLanes polylanes;

    //(2)Add find driving way algorithm
    //interface::PolyfitLane driving_way = this->FindDrivingWay(vehicle_state, lane_points);

    // (현재는 수정 없이 그대로 전달, 나중에 lane change 로직 추가 예정)
    interface::PolyfitLane driving_way_real = driving_way_raw;  // ← 새로운 변수 생성
    
    //[11.28 다훈 수정] BehaviorPlanning 함수 추가
    // ctx
    PlanningNode::BehaviorContext ctx = BehaviorPlanning(vehicle_state, mission);

    //[다훈 수정2] limit_speed 인자 추가
    //(3)Add velocity planning algorithm
    double reference_speed = VelocityPlanning(vehicle_state, lane_points, mission, driving_way_real, ctx);

    //->이거는 control에서 사용해야할거 같은데 나중에 옮기기 
    //interface::VehicleCommand vehicle_command;
    //vehicle_command.steering = steering;
    //vehicle_command.accel = 0.08; // 정지
    //vehicle_command.brake = 0.0;

    //////////////////////////////////////////////////
    
    //if (cfg_.use_manual_inputs == true) {
    //    p_vehicle_command_ = manual_input;
    //}

    //===================================================
    // Publish output
    //===================================================
    // (1) Publish vehicle command
    //p_vehicle_command_->publish(ros2_bridge::UpdateVehicleCommand(vehicle_command));

    // (2) Publish driving way
    p_driving_way_real_->publish(ros2_bridge::UpdatePolyfitLane(driving_way_real));

    //[다훈 수정] reference_speed publish
    std_msgs::msg::Float32 ref_msg;
    ref_msg.data = reference_speed;
    p_reference_speed_->publish(ref_msg);

    // (3) Publish polyfit lanes
    //p_poly_lanes_->publish(ros2_bridge::UpdatePolyfitLaneDataArray(polylanes));

    // (4) reference speed
    // Q. 이 부분 내가 velocity_planning에서 계산한 reference_speed를 퍼블리시 해서 lateral로 보내야하는데 어떻게 보내는거지? ros2_bridge 이용? -> bridge말고 std_msgs로 퍼블리시

}
//============================================
//BehaviorPlanning 함수 구현
// input: vehicle_state / mission
// output: BehaviorContext (현재 모드 정보)
//============================================
PlanningNode::BehaviorContext PlanningNode::BehaviorPlanning(const interface::VehicleState &vehicle_state, const interface::Mission &mission) {
    // ctx initialization
    BehaviorContext ctx;
    ctx.current_mode = DrivingMode::NORMAL_DRIVING; // 기본 모드 설정

    // 모든 object 돌면서 확인 
    for (const auto &obj : mission.objects) {
        //1) Global->Ego Coordinate 변환
        auto [x_rel, y_rel] = GlobalToEgoCoordinate(vehicle_state, obj.x, obj.y);

        //2) ego 앞쪽(+x) + 같은 lane(좌우 2m)만 고려 
        if (x_rel <= 0.0 || std::abs(y_rel) > 2.0) {
            continue; // 뒤쪽이거나 다른 lane에 있는 객체는 무시
        }

        //3) Dynamic object 중에서 TTC 계산
        if (obj.object_type == "Dynamic") {
            ctx.has_dynamic_object = true; // Dynamic Object (앞차) 존재
            
            // Dynamic object와 ego 차량의 상대 속도 계산 
            double v_rel = vehicle_state.velocity - obj.velocity; // 상대 속도
            if (v_rel > 0) { // 추월 상황 (ego가 더 빠름)
                // TTC 계산
                double ttc = x_rel / v_rel; // Time to Collision
                ctx.ttc_lead = std::min(ctx.ttc_lead, ttc); // 가장 작은 TTC 저장(for문 밖에서 쓸 수가 없어서 이거 진행)
                // ✅ 로그 추가
                RCLCPP_INFO(this->get_logger(), 
                    "[SCC] TTC: %.2f sec, x_rel: %.2f m, v_ego: %.2f m/s, v_lead: %.2f m/s",
                    ttc, x_rel, vehicle_state.velocity, obj.velocity);

                //double v_target;
                if (ttc < 3.0) { // TTC가 3초 미만인 경우에만 속도 정보 업데이트
                    ctx.v_lead = std::min(ctx.v_lead*0.5, obj.velocity); // 가장 가까운 앞차 속도 저장 (velocity planning함수에서 하고 싶었지만 ㅠ obj.velocity때문에 BehaviorPlanning에서 진행)
                }
                else if (ttc < 5.0) {
                    ctx.v_lead = std::min(ctx.v_lead*0.6, obj.velocity);
                }
                else if (ttc < 10.0) {
                    ctx.v_lead = std::min(ctx.v_lead*0.7, obj.velocity);
                }
                else if (ttc < 15.0) {
                    ctx.v_lead = std::min(ctx.v_lead*0.8, obj.velocity);
                }
                else {
                    ctx.v_lead = std::min(ctx.v_lead, obj.velocity);
                }
            }
        }

        //4) Static object 
        else if (obj.object_type == "Static") {
            ctx.has_static_object = true; // Static Object (장애물) 존재
            ctx.dist_static = std::min(ctx.dist_static, x_rel); // 가장 가까운 장애물 거리 저장(x_rel을 for문 밖에서 쓸 수가 없어서 이거 진행)
        }
    }

    //===================================
    //5) 모드 결정 로직 (1순위 static, 2순위 dynamic, 3순위 일반주행)
    //===================================
    // lane change 전환 thresholds = static_warn_dist 
    // scc 전환 thresholds = ttc_threshold
    const double static_warn_dist = 15.0; // 15m 이내에 static object 있으면 lane change
    const double ttc_threshold = 15.0; // 15초 이내에 충돌 예상이면 scc

    //모드 결정
    if (ctx.has_static_object && ctx.dist_static < static_warn_dist) {
        ctx.current_mode = DrivingMode::LANE_CHANGE;
    }
    else if (ctx.has_dynamic_object && ctx.ttc_lead < ttc_threshold) {
        ctx.current_mode = DrivingMode::SCC;
    }
    else {
        ctx.current_mode = DrivingMode::NORMAL_DRIVING;
    }

    return ctx;
    }

//============================================
// VelocityPlanning 
// input: vehicle_state / lane_points / limit_speed 
// output: driving_way / reference_speed(LongitudinalControl로 전달됨)
//============================================
//[다훈 수정3] limit_speed 인자 추가
double PlanningNode::VelocityPlanning(const interface::VehicleState &vehicle_state, const interface::Lane &lane_points, const interface::Mission &mission, const interface::PolyfitLane &driving_way_real, const BehaviorContext &ctx) {
    /**
     * @brief Plan the driving way and desired speed along the lane points
     * inputs: vehicle_state, lane_points, limit_speed
     * outputs: driving_way
     * Purpose: Generate a smooth driving path and set desired speed based on limit speed
     */
    
     //perception_node.cpp에서 구현한 FindDrivingWay 함수 driving_way 사용

    double v_ref = mission.speed_limit*0.5; // mission에서 직접 가져오기 (이게 v_lead)
    //=================================================
    //1) 곡률 기반 속도 제한 (v_kappa 계산)
    //=================================================
     //(1) PolyfitLaneData에서 a3, a2, a1, a0 가져옴 
    double a3 = driving_way_real.a3;
    double a2 = driving_way_real.a2;
    double a1 = driving_way_real.a1;
    double a0 = driving_way_real.a0;

    //(2) Kappa (곡률) 계산 및 속도 제한 적용 (ppt에서 k=2b=2*a2)
    double kappa = 2.0 * a2;

    //(3) 곡률 기반 속도 제한 v_kappa 계산 
    double eps = 1e-6; // 작은 값으로 나누기 방지
    double v_kappa;

    if(std::abs(kappa) < eps) {
        v_kappa = v_ref; // 곡률이 거의 0이면 그냥 목표 속도(limit_speed) 사용
    } else {
        v_kappa = std::sqrt(cfg_.param_max_lateral_accel / std::abs(kappa)); // v_kappa = sqrt(a_lat_max / |kappa|)
        v_kappa = std::min(v_kappa, v_ref); // 제한 속도 초과하지 않도록
    }

    //기본 속도 
    double reference_speed = std::min(v_ref, v_kappa);

    //=================================================
    // [11.28 다훈 수정] Mode 별 속도 결정
    //  - NORMAL_DRIVING: v_real 그대로 사용
    //  - SCC: 앞차 속도 동일
    //  - LANE_CHANGE: Static object와의 거리 고려해서 감속 및 lane change
    //=================================================
    // SCC 모드 
    if (ctx.current_mode == DrivingMode::SCC && ctx.has_dynamic_object) {
        //앞차 속도 동일하게
        reference_speed = std::min(reference_speed, ctx.v_lead);
    }
    // LANE_CHANGE 모드
    if (ctx.current_mode == DrivingMode::LANE_CHANGE && ctx.has_static_object) {
        // 일단 보류
    }
    else {
        // NORMAL_DRIVING 모드: v_real 그대로 사용
    }


    // obstacle 좌표계 global -> ego 변환 및 TTC 계산 (같은 lane에 있는 차량만 고려)
    // for (const auto& obj : mission.objects) {
    //     //[11.28 다훈 수정] 함수 불러와서 obstacle 좌표계 변환
    //     auto [x_rel, y_rel] = GlobalToEgoCoordinate(vehicle_state, obj.x, obj.y);
        
    //     if (obj.object_type == "Dynamic") {
    //         //
    //         // [11.27 다훈 수정] 같은 lane에 있는 object만 고려 (y_rel이 작고 x_rel이 양수인 경우)
    //         if (std::abs(y_rel)<2.0 && x_rel > 0) { // [부등호 < or <= (고민 중)]
    //             double v_rel = vehicle_state.velocity - obj.velocity; // 상대 속도
    //             if (v_rel > 0) { // 추월 상황 (ego가 더 빠름)
    //                 double ttc = x_rel / v_rel; // Time to Collision
    //                 // ttc를 더 잘게 나누어서 진행하자. 
    //                 if (ttc < 5.0) { // TTC가 임계값 이하인 경우 [cfg_.param_scc_ttc_threshold]
    //                     // 이하이면 속도 조정 필요 앞차와의 속도 동일하게 즉, (SCC reference speed = std::min(limit_speed, v_kappa, v_lead))
    //                     v_lead = std::min(v_lead, obj.velocity);
    //                     found_leading_vehicle = true;
    //                 }
    //             }
    //         }
    //     }
    // }
    //A-2 TTC 계산 (같은 lane에 있는 차량만 고려)
    // double reference_speed;
    // if (found_leading_vehicle) {
    //     reference_speed = std::min({limit_speed, v_kappa, v_lead});
    // } else {
    //     reference_speed = std::min(limit_speed, v_kappa);
    // }
    // //(4) 최종 목표 속도 설정 (limit_speed와 v_kappa 중 더 작은 값)
    // //double reference_speed = std::min(limit_speed, v_kappa);

    return reference_speed;
}

// GlobalToEgoCoordinate 함수 구현 (일단 보류) 따로 뺄까 그냥 velocity_planning에 넣을까?
std::pair<double, double> PlanningNode::GlobalToEgoCoordinate(const interface::VehicleState &vehicle_state, double obj_x_global, double obj_y_global) {
    // A-1 obstacle (global coordinate) -> ego 기준 coordinate 변환
    double dx = obj_x_global - vehicle_state.x;
    double dy = obj_y_global - vehicle_state.y;
    double cos_yaw = std::cos(vehicle_state.yaw);
    double sin_yaw = std::sin(vehicle_state.yaw);
    double x_rel = cos_yaw * dx + sin_yaw * dy;
    double y_rel = -sin_yaw * dx + cos_yaw * dy;

    return std::make_pair(x_rel, y_rel);
}

// TTC 함수 밖으로 빼기 
// LaneChange 함수 추가 (나중에 구현)

int main(int argc, char **argv) {
    std::string node_name = "planning_node";

    // Initialize node
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PlanningNode>(node_name));
    rclcpp::shutdown();
    return 0;
}