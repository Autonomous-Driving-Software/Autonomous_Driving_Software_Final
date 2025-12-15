// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from ad_msgs:msg/LanePointData.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__LANE_POINT_DATA__BUILDER_HPP_
#define AD_MSGS__MSG__DETAIL__LANE_POINT_DATA__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "ad_msgs/msg/detail/lane_point_data__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace ad_msgs
{

namespace msg
{

namespace builder
{

class Init_LanePointData_point
{
public:
  explicit Init_LanePointData_point(::ad_msgs::msg::LanePointData & msg)
  : msg_(msg)
  {}
  ::ad_msgs::msg::LanePointData point(::ad_msgs::msg::LanePointData::_point_type arg)
  {
    msg_.point = std::move(arg);
    return std::move(msg_);
  }

private:
  ::ad_msgs::msg::LanePointData msg_;
};

class Init_LanePointData_id
{
public:
  explicit Init_LanePointData_id(::ad_msgs::msg::LanePointData & msg)
  : msg_(msg)
  {}
  Init_LanePointData_point id(::ad_msgs::msg::LanePointData::_id_type arg)
  {
    msg_.id = std::move(arg);
    return Init_LanePointData_point(msg_);
  }

private:
  ::ad_msgs::msg::LanePointData msg_;
};

class Init_LanePointData_frame_id
{
public:
  Init_LanePointData_frame_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_LanePointData_id frame_id(::ad_msgs::msg::LanePointData::_frame_id_type arg)
  {
    msg_.frame_id = std::move(arg);
    return Init_LanePointData_id(msg_);
  }

private:
  ::ad_msgs::msg::LanePointData msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::ad_msgs::msg::LanePointData>()
{
  return ad_msgs::msg::builder::Init_LanePointData_frame_id();
}

}  // namespace ad_msgs

#endif  // AD_MSGS__MSG__DETAIL__LANE_POINT_DATA__BUILDER_HPP_
