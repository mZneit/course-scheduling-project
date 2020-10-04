# Course Scheduling Project
Course scheduling is a highly constrained satisfaction problem. The project aims to find a consistent schedule given a number of university courses subject to multiple constraints. A consistent schedule maps courses to a fixed number of rooms and timeslots where each course is assigned an instructor in a conflict-free manner.

The set of constraints is mainly divided into two categories – hard and soft constraints; the hard constraints should be satisfied at every instance (for instance, room and course conflicts), whereas the soft constraints can be violated as necessary. Soft constraints such as the professors’ preferences and the course offering throughout the day (morning/evening) are taken into consideration. The goal is to maximize the number of satisfied soft constraints (for instance, the instructor’s preferences) without introducing any conflicts.

## Team Contributions
Manal Zneit – implemented the algorithm and the code that solves the course scheduling CSP 

Jonathan Kelaty – devised a large real-world dataset drawn from Hunter College course offerings. He also implemented the visualization tool to demo the algorithm.

## Implementation
* The algorithm implemented is backtracking search with inference. For the large dataset, an interative approach was introduced.

* The consistent schedules that were the best candidates to satisfy most of the soft constraints were clustered in the first few iterations of the search.

* Random-restart hill climbing which is a local search algorithm was also implemented to randomize the input and avoid local minima. 

## Results

The algorithm generated a consistent schedule with maximized evaluation for a small and large datasets. The CSP satisfied all the proposed hard constraints with a delicate 
consideration of the soft constraints in an integrated use of algorithms to achieve the results in a feasible amount of time.
