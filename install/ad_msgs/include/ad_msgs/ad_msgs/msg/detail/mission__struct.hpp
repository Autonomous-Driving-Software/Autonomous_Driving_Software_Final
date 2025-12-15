// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from ad_msgs:msg/Mission.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__MISSION__STRUCT_HPP_
#define AD_MSGS__MSG__DETAIL__MISSION__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'objects'
#include "ad_msgs/msg/detail/mission_object__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__ad_msgs__msg__Mission __attribute__((deprecated))
#else
# define DEPRECATED__ad_msgs__msg__Mission __declspec(deprecated)
#endif

namespace ad_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Mission_
{
  using Type = Mission_<ContainerAllocator>;

  explicit Mission_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->road_condition = "";
      this->road_slope = "";
      this->speed_limit = 0.0;
      this->parking = false;
    }
  }

  explicit Mission_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : road_condition(_alloc),
    road_slope(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->road_condition = "";
      this->road_slope = "";
      this->speed_limit = 0.0;
      this->parking = false;
    }
  }

  // field types and members
  using _objects_type =
    std::vector<ad_msgs::msg::MissionObject_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<ad_msgs::msg::MissionObject_<ContainerAllocator>>>;
  _objects_type objects;
  using _road_condition_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _road_condition_type road_condition;
  using _road_slope_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _road_slope_type road_slope;
  using _speed_limit_type =
    double;
  _speed_limit_type speed_limit;
  using _parking_type =
    bool;
  _parking_type parking;

  // setters for named parameter idiom
  Type & set__objects(
    const std::vector<ad_msgs::msg::MissionObject_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<ad_msgs::msg::MissionObject_<ContainerAllocator>>> & _arg)
  {
    this->objects = _arg;
    return *this;
  }
  Type & set__road_condition(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->road_condition = _arg;
    return *this;
  }
  Type & set__road_slope(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->road_slope = _arg;
    return *this;
  }
  Type & set__speed_limit(
    const double & _arg)
  {
    this->speed_limit = _arg;
    return *this;
  }
  Type & set__parking(
    const bool & _arg)
  {
    this->parking = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    ad_msgs::msg::Mission_<ContainerAllocator> *;
  using ConstRawPtr =
    const ad_msgs::msg::Mission_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<ad_msgs::msg::Mission_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<ad_msgs::msg::Mission_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      ad_msgs::msg::Mission_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<ad_msgs::msg::Mission_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      ad_msgs::msg::Mission_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<ad_msgs::msg::Mission_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<ad_msgs::msg::Mission_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<ad_msgs::msg::Mission_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__ad_msgs__msg__Mission
    std::shared_ptr<ad_msgs::msg::Mission_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__ad_msgs__msg__Mission
    std::shared_ptr<ad_msgs::msg::Mission_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Mission_ & other) const
  {
    if (this->objects != other.objects) {
      return false;
    }
    if (this->road_condition != other.road_condition) {
      return false;
    }
    if (this->road_slope != other.road_slope) {
      return false;
    }
    if (this->speed_limit != other.speed_limit) {
      return false;
    }
    if (this->parking != other.parking) {
      return false;
    }
    return true;
  }
  bool operator!=(const Mission_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Mission_

// alias to use template instance with default allocator
using Mission =
  ad_msgs::msg::Mission_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace ad_msgs

#endif  // AD_MSGS__MSG__DETAIL__MISSION__STRUCT_HPP_
