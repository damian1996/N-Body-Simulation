# N-BODY SIMULATION #

### Authors ###
Damian Stachura

### INSTALLATION FOR UBUNTU 16.04 ###
### REQUIREMENTS ###
- **Install glfw3 and glm**
   * sudo apt-get update
   * sudo apt-get install libglm-dev glibglfw3-dev libglfw3
- **Install  CUDA 9.2**
   * Go to https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions
   * Go to subsection 2.6 and go to the download section.
   * Then choose sufficient options and finally you will choose deb(local).
   * Later click Downlad and enter commands from installation guide below
- **Thrust Library**
   * sudo apt-get update
   * sudo apt-get install libthrust-dev
### EXECUTE ###
- **Clone repository**
   * Move to your favourite directory and enter following command   
   * git clone https://github.com/damian1996/FlappyGil.git
- **Run program**
   Enter following commands to run program
   * cd N-Body-Simulation
   * cd Naive
   * make clean && make
   * ./out/main
    and have fun with it!
