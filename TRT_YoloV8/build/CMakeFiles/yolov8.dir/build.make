# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/liujinfu/Desktop/TRT_YoloV8

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/liujinfu/Desktop/TRT_YoloV8/build

# Include any dependencies generated for this target.
include CMakeFiles/yolov8.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/yolov8.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/yolov8.dir/flags.make

CMakeFiles/yolov8.dir/my_main.cpp.o: CMakeFiles/yolov8.dir/flags.make
CMakeFiles/yolov8.dir/my_main.cpp.o: ../my_main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liujinfu/Desktop/TRT_YoloV8/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/yolov8.dir/my_main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yolov8.dir/my_main.cpp.o -c /home/liujinfu/Desktop/TRT_YoloV8/my_main.cpp

CMakeFiles/yolov8.dir/my_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolov8.dir/my_main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liujinfu/Desktop/TRT_YoloV8/my_main.cpp > CMakeFiles/yolov8.dir/my_main.cpp.i

CMakeFiles/yolov8.dir/my_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolov8.dir/my_main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liujinfu/Desktop/TRT_YoloV8/my_main.cpp -o CMakeFiles/yolov8.dir/my_main.cpp.s

CMakeFiles/yolov8.dir/src/block.cpp.o: CMakeFiles/yolov8.dir/flags.make
CMakeFiles/yolov8.dir/src/block.cpp.o: ../src/block.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liujinfu/Desktop/TRT_YoloV8/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/yolov8.dir/src/block.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yolov8.dir/src/block.cpp.o -c /home/liujinfu/Desktop/TRT_YoloV8/src/block.cpp

CMakeFiles/yolov8.dir/src/block.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolov8.dir/src/block.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liujinfu/Desktop/TRT_YoloV8/src/block.cpp > CMakeFiles/yolov8.dir/src/block.cpp.i

CMakeFiles/yolov8.dir/src/block.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolov8.dir/src/block.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liujinfu/Desktop/TRT_YoloV8/src/block.cpp -o CMakeFiles/yolov8.dir/src/block.cpp.s

CMakeFiles/yolov8.dir/src/calibrator.cpp.o: CMakeFiles/yolov8.dir/flags.make
CMakeFiles/yolov8.dir/src/calibrator.cpp.o: ../src/calibrator.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liujinfu/Desktop/TRT_YoloV8/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/yolov8.dir/src/calibrator.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yolov8.dir/src/calibrator.cpp.o -c /home/liujinfu/Desktop/TRT_YoloV8/src/calibrator.cpp

CMakeFiles/yolov8.dir/src/calibrator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolov8.dir/src/calibrator.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liujinfu/Desktop/TRT_YoloV8/src/calibrator.cpp > CMakeFiles/yolov8.dir/src/calibrator.cpp.i

CMakeFiles/yolov8.dir/src/calibrator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolov8.dir/src/calibrator.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liujinfu/Desktop/TRT_YoloV8/src/calibrator.cpp -o CMakeFiles/yolov8.dir/src/calibrator.cpp.s

CMakeFiles/yolov8.dir/src/model.cpp.o: CMakeFiles/yolov8.dir/flags.make
CMakeFiles/yolov8.dir/src/model.cpp.o: ../src/model.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liujinfu/Desktop/TRT_YoloV8/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/yolov8.dir/src/model.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yolov8.dir/src/model.cpp.o -c /home/liujinfu/Desktop/TRT_YoloV8/src/model.cpp

CMakeFiles/yolov8.dir/src/model.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolov8.dir/src/model.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liujinfu/Desktop/TRT_YoloV8/src/model.cpp > CMakeFiles/yolov8.dir/src/model.cpp.i

CMakeFiles/yolov8.dir/src/model.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolov8.dir/src/model.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liujinfu/Desktop/TRT_YoloV8/src/model.cpp -o CMakeFiles/yolov8.dir/src/model.cpp.s

CMakeFiles/yolov8.dir/src/postprocess.cpp.o: CMakeFiles/yolov8.dir/flags.make
CMakeFiles/yolov8.dir/src/postprocess.cpp.o: ../src/postprocess.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liujinfu/Desktop/TRT_YoloV8/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/yolov8.dir/src/postprocess.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yolov8.dir/src/postprocess.cpp.o -c /home/liujinfu/Desktop/TRT_YoloV8/src/postprocess.cpp

CMakeFiles/yolov8.dir/src/postprocess.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolov8.dir/src/postprocess.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liujinfu/Desktop/TRT_YoloV8/src/postprocess.cpp > CMakeFiles/yolov8.dir/src/postprocess.cpp.i

CMakeFiles/yolov8.dir/src/postprocess.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolov8.dir/src/postprocess.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liujinfu/Desktop/TRT_YoloV8/src/postprocess.cpp -o CMakeFiles/yolov8.dir/src/postprocess.cpp.s

CMakeFiles/yolov8.dir/src/postprocess.cu.o: CMakeFiles/yolov8.dir/flags.make
CMakeFiles/yolov8.dir/src/postprocess.cu.o: ../src/postprocess.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liujinfu/Desktop/TRT_YoloV8/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CUDA object CMakeFiles/yolov8.dir/src/postprocess.cu.o"
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/liujinfu/Desktop/TRT_YoloV8/src/postprocess.cu -o CMakeFiles/yolov8.dir/src/postprocess.cu.o

CMakeFiles/yolov8.dir/src/postprocess.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/yolov8.dir/src/postprocess.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/yolov8.dir/src/postprocess.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/yolov8.dir/src/postprocess.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/yolov8.dir/src/preprocess.cu.o: CMakeFiles/yolov8.dir/flags.make
CMakeFiles/yolov8.dir/src/preprocess.cu.o: ../src/preprocess.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liujinfu/Desktop/TRT_YoloV8/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CUDA object CMakeFiles/yolov8.dir/src/preprocess.cu.o"
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/liujinfu/Desktop/TRT_YoloV8/src/preprocess.cu -o CMakeFiles/yolov8.dir/src/preprocess.cu.o

CMakeFiles/yolov8.dir/src/preprocess.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/yolov8.dir/src/preprocess.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/yolov8.dir/src/preprocess.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/yolov8.dir/src/preprocess.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target yolov8
yolov8_OBJECTS = \
"CMakeFiles/yolov8.dir/my_main.cpp.o" \
"CMakeFiles/yolov8.dir/src/block.cpp.o" \
"CMakeFiles/yolov8.dir/src/calibrator.cpp.o" \
"CMakeFiles/yolov8.dir/src/model.cpp.o" \
"CMakeFiles/yolov8.dir/src/postprocess.cpp.o" \
"CMakeFiles/yolov8.dir/src/postprocess.cu.o" \
"CMakeFiles/yolov8.dir/src/preprocess.cu.o"

# External object files for target yolov8
yolov8_EXTERNAL_OBJECTS =

yolov8: CMakeFiles/yolov8.dir/my_main.cpp.o
yolov8: CMakeFiles/yolov8.dir/src/block.cpp.o
yolov8: CMakeFiles/yolov8.dir/src/calibrator.cpp.o
yolov8: CMakeFiles/yolov8.dir/src/model.cpp.o
yolov8: CMakeFiles/yolov8.dir/src/postprocess.cpp.o
yolov8: CMakeFiles/yolov8.dir/src/postprocess.cu.o
yolov8: CMakeFiles/yolov8.dir/src/preprocess.cu.o
yolov8: CMakeFiles/yolov8.dir/build.make
yolov8: libmyplugins.so
yolov8: /usr/local/lib/libopencv_gapi.so.4.7.0
yolov8: /usr/local/lib/libopencv_highgui.so.4.7.0
yolov8: /usr/local/lib/libopencv_ml.so.4.7.0
yolov8: /usr/local/lib/libopencv_objdetect.so.4.7.0
yolov8: /usr/local/lib/libopencv_photo.so.4.7.0
yolov8: /usr/local/lib/libopencv_stitching.so.4.7.0
yolov8: /usr/local/lib/libopencv_video.so.4.7.0
yolov8: /usr/local/lib/libopencv_videoio.so.4.7.0
yolov8: /usr/local/lib/libopencv_imgcodecs.so.4.7.0
yolov8: /usr/local/lib/libopencv_dnn.so.4.7.0
yolov8: /usr/local/lib/libopencv_calib3d.so.4.7.0
yolov8: /usr/local/lib/libopencv_features2d.so.4.7.0
yolov8: /usr/local/lib/libopencv_flann.so.4.7.0
yolov8: /usr/local/lib/libopencv_imgproc.so.4.7.0
yolov8: /usr/local/lib/libopencv_core.so.4.7.0
yolov8: CMakeFiles/yolov8.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/liujinfu/Desktop/TRT_YoloV8/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX executable yolov8"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/yolov8.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/yolov8.dir/build: yolov8

.PHONY : CMakeFiles/yolov8.dir/build

CMakeFiles/yolov8.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/yolov8.dir/cmake_clean.cmake
.PHONY : CMakeFiles/yolov8.dir/clean

CMakeFiles/yolov8.dir/depend:
	cd /home/liujinfu/Desktop/TRT_YoloV8/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liujinfu/Desktop/TRT_YoloV8 /home/liujinfu/Desktop/TRT_YoloV8 /home/liujinfu/Desktop/TRT_YoloV8/build /home/liujinfu/Desktop/TRT_YoloV8/build /home/liujinfu/Desktop/TRT_YoloV8/build/CMakeFiles/yolov8.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/yolov8.dir/depend

