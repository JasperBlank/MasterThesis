Raw_input shows the raw input file as it comes out of the camera

Life_needle_protocol shows what the protocol does. 
It first does edge detection then it identifies the needle as 2 close edges (2-22 pixels)
 in the same direciton (less then 12 degrees different)
 coming from off screen (both edges must have one side near the border).

Based on these edges it identifies the tip of the needle as the place where both edges end

Then it detects the dot based on color. It detects a blob of red pixels and calculates the average

Currently the Automatic-dot-targeting as shown in the video is not robust, and sometimes goes off into a wrong direction



Next to this all motors can also be controlled manually shown in manual_control.mp4. Manual_control_outside shows the same from the outside perspective.
