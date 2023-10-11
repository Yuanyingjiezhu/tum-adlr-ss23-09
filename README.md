# Trajectory planning with moving obstacle

<div style="text-align: center">
<img src="media/379565c6cb93534ab773dba81eb443f.png" width="600"/>
</div>

This project aims to first find an appropriate way to encode dynamic environment with the help of BPS method due to its powerful performance. And then with the encoded information
we make trajectory planning of drones using supervised learning. The target is achieved by adjusted A* Algorithm.

## Dataset
<div style="text-align: center">
<img src="media/demo_syn_room.gif" width="600"/>
</div>
To evaluate a pretrained model or train a new model from scratch, you have to obtain the respective dataset.
In this paper, we consider the following dataset:
we generate a complex dynamic environment which contains 16 moving obstacles. The size of each obstacle is randomly generated, and to better simulate the movement in real world, we add some noise in it, such as time-varying velocity and rotation speed.




