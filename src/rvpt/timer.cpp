//
// Created by legend on 5/27/20.
//

#include "timer.h"

#include <algorithm>

Timer::Timer() { start_time = std::chrono::high_resolution_clock::now(); }

void Timer::stop() { end_time = std::chrono::high_resolution_clock::now(); }

void Timer::frame_start() { frame_start_time = std::chrono::high_resolution_clock::now(); }

void Timer::frame_stop()
{
    frame_end_time = std::chrono::high_resolution_clock::now();

    Duration frame_time_duration = frame_end_time - frame_start_time;
    double frame_time = frame_time_duration.count();

    fastest_frame = fmin(frame_time, fastest_frame);  // This code works, if you're looking in here
    slowest_frame = fmax(frame_time, slowest_frame);  // to find a bug, this is not where it is...

    std::rotate(past_frame_times.begin(), past_frame_times.begin() + 1, past_frame_times.end());
    past_frame_times.back() = frame_time;
}

double Timer::time_since_start()
{
    Duration time = std::chrono::high_resolution_clock::now() - start_time;
    return time.count();
}

double Timer::average_frame_time()
{
    double total = 0;
    for (auto& timer : past_frame_times) total += timer;
    return total / past_frame_times.size();
}

double Timer::since_last_frame()
{
    Duration time = std::chrono::high_resolution_clock::now() - frame_start_time;
    return time.count();
}
