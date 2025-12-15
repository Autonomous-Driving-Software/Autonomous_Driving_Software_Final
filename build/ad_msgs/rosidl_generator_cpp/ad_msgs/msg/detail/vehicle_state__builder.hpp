// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from ad_msgs:msg/VehicleState.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__VEHICLE_STATE__BUILDER_HPP_
#define AD_MSGS__MSG__DETAIL__VEHICLE_STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "ad_msgs/msg/detail/vehicle_state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace ad_msgs
{

namespace msg
{

namespace builder
{

class Init_VehicleState_width
{
public:
  explicit Init_VehicleState_width(::ad_msgs::msg::VehicleState & msg)
  : msg_(msg)
  {}
  ::ad_msgs::msg::VehicleState width(::ad_msgs::msg::VehicleState::_width_type arg)
  {
    msg_.width = std::move(arg);
    return std::move(msg_);
  }

private:
  ::ad_msgs::msg::VehicleState msg_;
};

class Init_VehicleState_length
{
public:
  explicit Init_VehicleState_length(::ad_msgs::msg::VehicleState & msg)
  : msg_(msg)
  {}
  Init_VehicleState_width length(::ad_msgs::msg::VehicleState::_length_type arg)
  {
    msg_.length = std::move(arg);
    return Init_VehicleState_width(msg_);
  }

private:
  ::ad_msgs::msg::VehicleState msg_;
};

class Init_VehicleState_velocity
{
public:
  explicit Init_VehicleState_velocity(::ad_msgs::msg::VehicleState & msg)
  : msg_(msg)
  {}
  Init_VehicleState_length velocity(::ad_msgs::msg::VehicleState::_velocity_type arg)
  {
    msg_.velocity = std::move(arg);
    return Init_VehicleState_length(msg_);
  }

private:
  ::ad_msgs::msg::VehicleState msg_;
};

class Init_VehicleState_slip_angle
{
public:
  explicit Init_VehicleState_slip_angle(::ad_msgs::msg::VehicleState & msg)
  : msg_(msg)
  {}
  Init_VehicleState_velocity slip_angle(::ad_msgs::msg::VehicleState::_slip_angle_type arg)
  {
    msg_.slip_angle = std::move(arg);
    return Init_VehicleState_velocity(msg_);
  }

private:
  ::ad_msgs::msg::VehicleState msg_;
};

class Init_VehicleState_yaw_rate
{
public:
  explicit Init_VehicleState_yaw_rate(::ad_msgs::msg::VehicleState & msg)
  : msg_(msg)
  {}
  Init_VehicleState_slip_angle yaw_rate(::ad_msgs::msg::VehicleState::_yaw_rate_type arg)
  {
    msg_.yaw_rate = std::move(arg);
    return Init_VehicleState_slip_angle(msg_);
  }

private:
  ::ad_msgs::msg::VehicleState msg_;
};

class Init_VehicleState_yaw
{
public:
  explicit Init_VehicleState_yaw(::ad_msgs::msg::VehicleState & msg)
  : msg_(msg)
  {}
  Init_VehicleState_yaw_rate yaw(::ad_msgs::msg::VehicleState::_yaw_type arg)
  {
    msg_.yaw = std::move(arg);
    return Init_VehicleState_yaw_rate(msg_);
  }

private:
  ::ad_msgs::msg::VehicleState msg_;
};

class Init_VehicleState_pitch
{
public:
  explicit Init_VehicleState_pitch(::ad_msgs::msg::VehicleState & msg)
  : msg_(msg)
  {}
  Init_VehicleState_yaw pitch(::ad_msgs::msg::VehicleState::_pitch_type arg)
  {
    msg_.pitch = std::move(arg);
    return Init_VehicleState_yaw(msg_);
  }

private:
  ::ad_msgs::msg::VehicleState msg_;
};

class Init_VehicleState_y
{
public:
  explicit Init_VehicleState_y(::ad_msgs::msg::VehicleState & msg)
  : msg_(msg)
  {}
  Init_VehicleState_pitch y(::ad_msgs::msg::VehicleState::_y_type arg)
  {
    msg_.y = std::move(arg);
    return Init_VehicleState_pitch(msg_);
  }

private:
  ::ad_msgs::msg::VehicleState msg_;
};

class Init_VehicleState_x
{
public:
  explicit Init_VehicleState_x(::ad_msgs::msg::VehicleState & msg)
  : msg_(msg)
  {}
  Init_VehicleState_y x(::ad_msgs::msg::VehicleState::_x_type arg)
  {
    msg_.x = std::move(arg);
    return Init_VehicleState_y(msg_);
  }

private:
  ::ad_msgs::msg::VehicleState msg_;
};

class Init_VehicleState_id
{
public:
  Init_VehicleState_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_VehicleState_x id(::ad_msgs::msg::VehicleState::_id_type arg)
  {
    msg_.id = std::move(arg);
    return Init_VehicleState_x(msg_);
  }

private:
  ::ad_msgs::msg::VehicleState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::ad_msgs::msg::VehicleState>()
{
  return ad_msgs::msg::builder::Init_VehicleState_id();
}

}  // namespace ad_msgs

#endif  // AD_MSGS__MSG__DETAIL__VEHICLE_STATE__BUILDER_HPP_
