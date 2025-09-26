#ifndef TRIEX_PLATFORM
#define TRIEX_PLATFORM

#include <stdbool.h>
#include <stdint.h>

void platform_create_window(const char* title, size_t width, size_t height);
bool platform_window_handle_events();
bool platform_still_running();
void platform_sleep(size_t milis);
bool platform_resize_window_callback(bool minimized);
uint64_t platform_get_time();
void platform_set_mouse_position(size_t x, size_t y);
void platform_enable_fullscreen();
void platform_disable_fullscreen();
bool platform_drag_and_drop_available();
char** platform_get_drag_and_drop_files(int* count);
void platform_release_drag_and_drop(char** files, int count);

extern bool platform_window_minimized;
#endif