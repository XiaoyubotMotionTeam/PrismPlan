# A single collision object: one or more solid primitives, each with a pose.
# Non-box primitives are over-approximated to their enclosing oriented box by
# the planner node, matching the substrate's sphere-vs-OBB collision path.

uint8 ADD=0
uint8 REMOVE=1

string id
shape_msgs/SolidPrimitive[] primitives
geometry_msgs/Pose[] primitive_poses
uint8 operation
