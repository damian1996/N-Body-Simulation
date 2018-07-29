# N-BODY SIMULATION #

### Authors ###
**Damian Stachura**

### INSTALLATION FOR UBUNTU 16.04 ###
### REQUIREMENTS ###
- **Install glfw3 and glm**\
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;` sudo apt-get update `\
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;` sudo apt-get install libglm-dev glibglfw3-dev libglfw3 `
- **Install  CUDA 9.2**\
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Go to [CUDA installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions)**\
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Go to subsection 2.6 and go to the download section.**\
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Then choose sufficient options and finally you will choose deb(local).**\
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Later click Downlad and enter commands from installation guide below**
- **Thrust Library**\
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;` sudo apt-get update `\
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;` sudo apt-get install libthrust-dev `
### EXECUTE ###
- **Clone repository**\
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Move to your favourite directory and enter following command**\
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;` git clone https://github.com/damian1996/FlappyGil.git `
- **Run program**\
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Enter following commands to run program**\
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;` cd N-Body-Simulation `\
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;` cd Naive `\
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;` make clean && make `\
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;` ./out/main `\
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**and have fun with it!**
