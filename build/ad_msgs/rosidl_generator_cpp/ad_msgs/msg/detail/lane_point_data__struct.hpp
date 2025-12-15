// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from ad_msgs:msg/LanePointData.idl
// generated code does not contain a copyright notice

#ifndef AD_MSGS__MSG__DETAIL__LANE_POINT_DATA__STRUCT_HPP_
#define AD_MSGS__MSG__DETAIL__LANE_POINT_DATA__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'point'
#include "geometry_msgs/msg/detail/point__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__ad_msgs__msg__LanePointData __attribute__((deprecated))
#else
# define DEPRECATED__ad_msgs__msg__LanePointData __declspec(deprecated)
#endif

namespace ad_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct LanePointData_
{
  using Type = LanePointData_<ContainerAllocator>;

  explicit LanePointData_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->frame_id = "";
      this->id = "";
    }
  }

  explicit LanePointData_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : frame_id(_alloc),
    id(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->frame_id = "";
      this->id = "";
    }
  }

  // field types and members
  using _frame_id_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _frame_id_type frame_id;
  using _id_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _id_type id;
  using _point_type =
    std::vector<geometry_msgs::msg::Point_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<geometry_msgs::msg::Point_<ContainerAllocator>>>;
  _point_type point;

  // setters for named parameter idiom
  Type & set__frame_id(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->frame_id = _arg;
    return *this;
  }
  Type & set__id(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->id = _arg;
    return *this;
  }
  Type & set__point(
    const std::vector<geometry_msgs::msg::Point_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<geometry_msgs::msg::Point_<ContainerAllocator>>> & _arg)
  {
    this->point = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    ad_msgs::msg::LanePointData_<ContainerAllocator> *;
  using ConstRawPtr =
    const ad_msgs::msg::LanePointData_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<ad_msgs::msg::LanePointData_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<ad_msgs::msg::LanePointData_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      ad_msgs::msg::LanePointData_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<ad_msgs::msg::LanePointData_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      ad_msgs::msg::LanePointData_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<ad_msgs::msg::LanePointData_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<ad_msgs::msg::LanePointData_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<ad_msgs::msg::LanePointData_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__ad_msgs__msg__LanePointData
    std::shared_ptr<ad_msgs::msg::LanePointData_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__ad_msgs__msg__LanePointData
    std::shared_ptr<ad_msgs::msg::LanePointData_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const LanePointData_ & other) const
  {
    if (this->frame_id != other.frame_id) {
      return false;
    }
    if (this->id != other.id) {
      return false;
    }
    if (this->point != other.point) {
      return false;
    }
    return true;
  }
  bool operator!=(const LanePointData_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct LanePointData_

// alias to use template instance with default allocator
using LanePointData =
  ad_msgs::msg::LanePointData_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace ad_msgs

#endif  // AD_MSGS__MSG__DETAIL__LANE_POINT_DATA__STRUCT_HPP_
