//
// Created by legend on 5/27/20.
//

#pragma once

#include <chrono>
#include <cmath>
#include <array>
#include <numeric>

class Timer
{
public:
    Timer();

    void stop();
    void frame_start();
    void frame_stop();

    double time_since_start();
    double average_frame_time();
    double since_last_frame();

    double fastest_frame = std::numeric_limits<double>::max();
    double slowest_frame = std::numeric_limits<double>::min();
    std::array<double, 50> past_frame_times{};

private:
    using Duration = std::chrono::duration<double, std::ratio<1, 1>>;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> frame_start_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> frame_end_time;
};
