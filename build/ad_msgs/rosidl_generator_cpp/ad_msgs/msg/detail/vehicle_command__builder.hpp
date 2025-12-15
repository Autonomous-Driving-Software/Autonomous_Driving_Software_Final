// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from ad_msgs:msg/VehicleCommand.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__VEHICLE_COMMAND__BUILDER_HPP_
#define AD_MSGS__MSG__DETAIL__VEHICLE_COMMAND__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "ad_msgs/msg/detail/vehicle_command__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace ad_msgs
{

namespace msg
{

namespace builder
{

class Init_VehicleCommand_brake
{
public:
  explicit Init_VehicleCommand_brake(::ad_msgs::msg::VehicleCommand & msg)
  : msg_(msg)
  {}
  ::ad_msgs::msg::VehicleCommand brake(::ad_msgs::msg::VehicleCommand::_brake_type arg)
  {
    msg_.brake = std::move(arg);
    return std::move(msg_);
  }

private:
  ::ad_msgs::msg::VehicleCommand msg_;
};

class Init_VehicleCommand_accel
{
public:
  explicit Init_VehicleCommand_accel(::ad_msgs::msg::VehicleCommand & msg)
  : msg_(msg)
  {}
  Init_VehicleCommand_brake accel(::ad_msgs::msg::VehicleCommand::_accel_type arg)
  {
    msg_.accel = std::move(arg);
    return Init_VehicleCommand_brake(msg_);
  }

private:
  ::ad_msgs::msg::VehicleCommand msg_;
};

class Init_VehicleCommand_steering
{
public:
  Init_VehicleCommand_steering()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_VehicleCommand_accel steering(::ad_msgs::msg::VehicleCommand::_steering_type arg)
  {
    msg_.steering = std::move(arg);
    return Init_VehicleCommand_accel(msg_);
  }

private:
  ::ad_msgs::msg::VehicleCommand msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::ad_msgs::msg::VehicleCommand>()
{
  return ad_msgs::msg::builder::Init_VehicleCommand_steering();
}

}  // namespace ad_msgs

#endif  // AD_MSGS__MSG__DETAIL__VEHICLE_COMMAND__BUILDER_HPP_
