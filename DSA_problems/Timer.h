#pragma once

//#include <iostream>
#include <chrono>
#include <thread>

enum type {
	nano,
	micro,
	mili,
	sec,
};

class Timer
{
public:
	float time = 0.0f;
	
	std::chrono::system_clock::time_point start;
	std::chrono::system_clock::time_point end;
	
	void startClock()
	{
		start = std::chrono::system_clock::now();
	}

	float stopClock(type timeMode)
	{
		end = std::chrono::system_clock::now();
		float duration = 0.0f;
		if (timeMode == sec)
			duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

		time = duration;

		return duration;
	}
};