// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from ad_msgs:msg/PolyfitLaneData.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__POLYFIT_LANE_DATA__BUILDER_HPP_
#define AD_MSGS__MSG__DETAIL__POLYFIT_LANE_DATA__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "ad_msgs/msg/detail/polyfit_lane_data__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace ad_msgs
{

namespace msg
{

namespace builder
{

class Init_PolyfitLaneData_a3
{
public:
  explicit Init_PolyfitLaneData_a3(::ad_msgs::msg::PolyfitLaneData & msg)
  : msg_(msg)
  {}
  ::ad_msgs::msg::PolyfitLaneData a3(::ad_msgs::msg::PolyfitLaneData::_a3_type arg)
  {
    msg_.a3 = std::move(arg);
    return std::move(msg_);
  }

private:
  ::ad_msgs::msg::PolyfitLaneData msg_;
};

class Init_PolyfitLaneData_a2
{
public:
  explicit Init_PolyfitLaneData_a2(::ad_msgs::msg::PolyfitLaneData & msg)
  : msg_(msg)
  {}
  Init_PolyfitLaneData_a3 a2(::ad_msgs::msg::PolyfitLaneData::_a2_type arg)
  {
    msg_.a2 = std::move(arg);
    return Init_PolyfitLaneData_a3(msg_);
  }

private:
  ::ad_msgs::msg::PolyfitLaneData msg_;
};

class Init_PolyfitLaneData_a1
{
public:
  explicit Init_PolyfitLaneData_a1(::ad_msgs::msg::PolyfitLaneData & msg)
  : msg_(msg)
  {}
  Init_PolyfitLaneData_a2 a1(::ad_msgs::msg::PolyfitLaneData::_a1_type arg)
  {
    msg_.a1 = std::move(arg);
    return Init_PolyfitLaneData_a2(msg_);
  }

private:
  ::ad_msgs::msg::PolyfitLaneData msg_;
};

class Init_PolyfitLaneData_a0
{
public:
  explicit Init_PolyfitLaneData_a0(::ad_msgs::msg::PolyfitLaneData & msg)
  : msg_(msg)
  {}
  Init_PolyfitLaneData_a1 a0(::ad_msgs::msg::PolyfitLaneData::_a0_type arg)
  {
    msg_.a0 = std::move(arg);
    return Init_PolyfitLaneData_a1(msg_);
  }

private:
  ::ad_msgs::msg::PolyfitLaneData msg_;
};

class Init_PolyfitLaneData_id
{
public:
  explicit Init_PolyfitLaneData_id(::ad_msgs::msg::PolyfitLaneData & msg)
  : msg_(msg)
  {}
  Init_PolyfitLaneData_a0 id(::ad_msgs::msg::PolyfitLaneData::_id_type arg)
  {
    msg_.id = std::move(arg);
    return Init_PolyfitLaneData_a0(msg_);
  }

private:
  ::ad_msgs::msg::PolyfitLaneData msg_;
};

class Init_PolyfitLaneData_frame_id
{
public:
  Init_PolyfitLaneData_frame_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PolyfitLaneData_id frame_id(::ad_msgs::msg::PolyfitLaneData::_frame_id_type arg)
  {
    msg_.frame_id = std::move(arg);
    return Init_PolyfitLaneData_id(msg_);
  }

private:
  ::ad_msgs::msg::PolyfitLaneData msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::ad_msgs::msg::PolyfitLaneData>()
{
  return ad_msgs::msg::builder::Init_PolyfitLaneData_frame_id();
}

}  // namespace ad_msgs

#endif  // AD_MSGS__MSG__DETAIL__POLYFIT_LANE_DATA__BUILDER_HPP_
