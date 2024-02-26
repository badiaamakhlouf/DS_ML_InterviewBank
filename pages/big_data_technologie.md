## 29- Distributed computing versus parallel computing ? 

- **Parallel computing:**
    - Allows breaking down a large computational task into smaller subtasks that can be executed simultaneously.
    - Subtasks execution is done simultaneously in parallel using multiple processors or cores within a single machine.
    - Characteristics: Shared Memory, Data Sharing, Single Machine (with multiple processors), lower communication overhead due to accessing shared memory directly. 
    - Applications : problems that can be divided into independent subtasks such as image processing, numerical simulations, scientific computing.
- **Distributed computing:**
    - Distributed computing divides a single task between multiple computers (nodes) to achieve a common goal.
    - Each computer used in distributed computing has its own processor.
    - Machines are often connected over a network, to work together on a computational task
    - Characteristics: multiple machines, communication over network, data distribution, designed with fault tolerance mechanisms since individual nodes may fail
    - Applications :
        - Large-scale data processing (e.g., big data analytics).
        - Web services, cloud computing, and distributed databases.
        - Solving problems that require the coordination of multiple machines.
        
- The choice between them depends on the nature of the problem, scale requirements, and communication considerations.
