# The full collision world: the planner node replaces its obstacle set with the
# objects listed here each time this message arrives.

std_msgs/Header header
gpu_planning_msgs/CollisionObject[] objects
