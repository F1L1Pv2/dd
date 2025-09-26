#include <X11/X.h>
#define XK_LATIN1
#define XK_MISCELLANY
#include <X11/keysymdef.h>
#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/extensions/Xrandr.h>
#include <X11/Xutil.h>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/stat.h>
#include <stdint.h>
#include <sys/time.h>
#include <string.h>

#include "platform.h"
#include "platform_globals.h"
#include "input.h"
#include "nob.h"

void platform_fill_keycode_lookup_table();

static bool running = false;
static Atom wmDeleteWindow;
static char** dropped_files = NULL;
static int dropped_file_count = 0;
static int BUTTONS_KEYCODE_OFFSET = 250;
Window window;
Display* display;

// Xdnd atoms
static Atom XdndAwareAtom = 0;
static Atom XdndEnterAtom = 0;
static Atom XdndPositionAtom = 0;
static Atom XdndStatusAtom = 0;
static Atom XdndActionCopyAtom = 0;
static Atom XdndDropAtom = 0;
static Atom XdndSelectionAtom = 0;
static Atom XdndFinishedAtom = 0;
static Atom UriListAtom = 0;

// Xdnd state
static Window drag_source = 0;
static Time drop_time = 0;

static void parse_uri_list(const char* data) {
    // Free previous drop data
    for (int i = 0; i < dropped_file_count; i++) free((void*)dropped_files[i]);
    free(dropped_files);
    dropped_files = NULL;
    dropped_file_count = 0;

    // Parse new URIs
    const char* start = data;
    while (*start) {
        while (*start == '\r' || *start == '\n') start++;
        const char* end = strchr(start, '\n');
        if (!end) end = start + strlen(start);

        if (strncmp(start, "file://", 7) == 0) {
            // Skip "file://" and trim trailing whitespace
            const char* path_start = start + 7;
            size_t length = end - path_start;
            
            // Strip trailing carriage return if exists
            if (length > 0 && path_start[length-1] == '\r') length--;
            
            // Copy path
            char* path = malloc(length + 1);
            strncpy(path, path_start, length);
            path[length] = '\0';
            
            // Add to list
            dropped_files = realloc(dropped_files, sizeof(char*) * (dropped_file_count + 1));
            dropped_files[dropped_file_count++] = path;
        }
        start = end;
    }
}

void platform_create_window(const char* title, size_t width, size_t height) {
    display = XOpenDisplay(NULL);
    window = XCreateSimpleWindow(display, DefaultRootWindow(display), 10, 10, width, height, 0, 0, 0);
    
    long event_mask = ExposureMask | KeyPressMask | KeyReleaseMask | ButtonPressMask | 
                     ButtonReleaseMask | StructureNotifyMask | PropertyChangeMask;
    XSelectInput(display, window, event_mask);
    XMapWindow(display, window);

    wmDeleteWindow = XInternAtom(display, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(display, window, &wmDeleteWindow, 1);
    XStoreName(display, window, title);

    // Initialize Xdnd atoms
    XdndAwareAtom = XInternAtom(display, "XdndAware", False);
    XdndEnterAtom = XInternAtom(display, "XdndEnter", False);
    XdndPositionAtom = XInternAtom(display, "XdndPosition", False);
    XdndStatusAtom = XInternAtom(display, "XdndStatus", False);
    XdndActionCopyAtom = XInternAtom(display, "XdndActionCopy", False);
    XdndDropAtom = XInternAtom(display, "XdndDrop", False);
    XdndSelectionAtom = XInternAtom(display, "XdndSelection", False);
    XdndFinishedAtom = XInternAtom(display, "XdndFinished", False);
    UriListAtom = XInternAtom(display, "text/uri-list", False);

    // Register as Xdnd aware
    unsigned long version = 5;
    XChangeProperty(display, window, XdndAwareAtom, XA_ATOM, 32, 
                   PropModeReplace, (unsigned char*)&version, 1);

    running = true;
    platform_fill_keycode_lookup_table();
}

bool platform_window_handle_events() {
    Window root, child;
    int root_x, root_y, win_x, win_y;
    unsigned int mask_return;
    XQueryPointer(display, window, &root, &child, &root_x, &root_y, &win_x, &win_y, &mask_return);

    input.scroll = 0;
    input.mouse_x = win_x;
    input.mouse_y = win_y;

    for(int i = 0; i < NOB_ARRAY_LEN(input.keys); i++) {
        input.keys[i].justPressed = 0;
        input.keys[i].justReleased = 0;
    }

    while (XPending(display)) {
        XEvent event;
        XNextEvent(display, &event);

        switch(event.type) {
            case ConfigureNotify: 
                if (event.xconfigure.width != 0 && event.xconfigure.height != 0) 
                    if(!platform_resize_window_callback(false)) return false;
                break;

            case KeyPress: case KeyRelease: {
                bool isDown = event.type == KeyPress;
                KeyCodeID keyCode = KeyCodeLookupTable[event.xkey.keycode];
                Key* key = &input.keys[keyCode];
                key->isDown = isDown;
                if(key->oldIsDown == !isDown) {
                    if(isDown) key->justPressed = 1;
                    else key->justReleased = 1;
                }
                key->oldIsDown = isDown;
                break;
            }

            case ButtonPress: case ButtonRelease: {
                bool isDown = event.type == ButtonPress;
                bool scroll = false;
                if(isDown) {
                    switch(event.xbutton.button) {
                        case Button4: {input.scroll = 120; scroll = true;} break;
                        case Button5: {input.scroll = -120; scroll = true;} break;
                    }
                }
                if(scroll) break;
                KeyCodeID keyCode = KeyCodeLookupTable[BUTTONS_KEYCODE_OFFSET + event.xbutton.button];
                Key* key = &input.keys[keyCode];
                key->isDown = isDown;
                if(key->oldIsDown == !isDown) {
                    if(isDown) key->justPressed = 1;
                    else key->justReleased = 1;
                }
                key->oldIsDown = isDown;
                break;
            }

            case ClientMessage: {
                Atom wmProtocols = XInternAtom(display, "WM_PROTOCOLS", False);

                if (event.xclient.message_type == XdndEnterAtom) {
                    // Get source window and version
                    drag_source = event.xclient.data.l[0];
                } 
                else if (event.xclient.message_type == XdndPositionAtom) {
                    // Reply that we accept the drop
                    drag_source = event.xclient.data.l[0];
                    
                    XEvent response = {0};
                    response.xclient.type = ClientMessage;
                    response.xclient.message_type = XdndStatusAtom;
                    response.xclient.display = display;
                    response.xclient.window = drag_source;
                    response.xclient.format = 32;
                    response.xclient.data.l[0] = window;
                    response.xclient.data.l[1] = 1; // Accept with no rectangle
                    response.xclient.data.l[2] = 0; // Empty rectangle
                    response.xclient.data.l[3] = 0;
                    response.xclient.data.l[4] = XdndActionCopyAtom;
                    
                    XSendEvent(display, drag_source, False, NoEventMask, &response);
                }
                else if (event.xclient.message_type == XdndDropAtom) {
                    drag_source = event.xclient.data.l[0];
                    drop_time = event.xclient.data.l[2];
                    XConvertSelection(display, XdndSelectionAtom, UriListAtom, 
                                     XdndSelectionAtom, window, drop_time);
                } 
                else if (event.xclient.message_type == wmProtocols && 
                         (Atom)event.xclient.data.l[0] == wmDeleteWindow) {
                    running = false;
                }
                break;
            }

            case SelectionNotify: 
                if (event.xselection.property && 
                    event.xselection.selection == XdndSelectionAtom &&
                    event.xselection.target == UriListAtom) {
                    
                    Atom actual_type;
                    int actual_format;
                    unsigned long nitems, bytes_after;
                    unsigned char* data = NULL;

                    if (XGetWindowProperty(display, window, event.xselection.property, 0, (~0L),
                                     False, AnyPropertyType, &actual_type, &actual_format,
                                     &nitems, &bytes_after, &data) == Success) {
                        if (data) {
                            parse_uri_list((const char*)data);
                            XFree(data);
                        }
                    }
                    
                    // Notify source we're done
                    if (drag_source) {
                        XEvent finished = {0};
                        finished.xclient.type = ClientMessage;
                        finished.xclient.message_type = XdndFinishedAtom;
                        finished.xclient.display = display;
                        finished.xclient.window = drag_source;
                        finished.xclient.format = 32;
                        finished.xclient.data.l[0] = window;
                        finished.xclient.data.l[1] = (data != NULL) ? 1 : 0;
                        finished.xclient.data.l[2] = None;
                        
                        XSendEvent(display, drag_source, False, NoEventMask, &finished);
                        drag_source = 0;
                    }
                    
                    // Delete the property
                    XDeleteProperty(display, window, event.xselection.property);
                }
                break;
        }
    }
    return true;
}

bool platform_still_running() { return running; }
void platform_sleep(size_t milis) { usleep(milis*1000); }

uint64_t platform_get_time() {
    #ifdef CLOCK_REALTIME
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        return (uint64_t)ts.tv_sec * 1000 + (ts.tv_nsec / 1000000);
    #else
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return ((uint64_t)tv.tv_sec * 1000 + (tv.tv_usec / 1000));
    #endif
}

void platform_fill_keycode_lookup_table() {
    // Alphabet keys
    for (char c = 'A'; c <= 'Z'; c++) {
        KeyCodeLookupTable[XKeysymToKeycode(display, XK_A + (c - 'A'))] = KEY_A + (c - 'A');
        KeyCodeLookupTable[XKeysymToKeycode(display, XK_a + (c - 'A'))] = KEY_A + (c - 'A');
    }

    // Number keys
    for (char c = '0'; c <= '9'; c++) {
        KeyCodeLookupTable[XKeysymToKeycode(display, XK_0 + (c - '0'))] = KEY_0 + (c - '0');
    }

    // Special keys
    struct { KeySym sym; KeyCodeID code; } keys[] = {
        {XK_space, KEY_SPACE}, {XK_grave, KEY_TICK}, {XK_minus, KEY_MINUS},
        {XK_equal, KEY_EQUAL}, {XK_bracketleft, KEY_LEFT_BRACKET}, {XK_bracketright, KEY_RIGHT_BRACKET},
        {XK_semicolon, KEY_SEMICOLON}, {XK_quotedbl, KEY_QUOTE}, {XK_comma, KEY_COMMA},
        {XK_period, KEY_PERIOD}, {XK_slash, KEY_FORWARD_SLASH}, {XK_backslash, KEY_BACKWARD_SLASH},
        {XK_Tab, KEY_TAB}, {XK_Escape, KEY_ESCAPE}, {XK_Pause, KEY_PAUSE},
        {XK_Up, KEY_UP}, {XK_Down, KEY_DOWN}, {XK_Left, KEY_LEFT}, {XK_Right, KEY_RIGHT},
        {XK_BackSpace, KEY_BACKSPACE}, {XK_Return, KEY_RETURN}, {XK_Delete, KEY_DELETE},
        {XK_Insert, KEY_INSERT}, {XK_Home, KEY_HOME}, {XK_End, KEY_END},
        {XK_Page_Up, KEY_PAGE_UP}, {XK_Page_Down, KEY_PAGE_DOWN}, {XK_Caps_Lock, KEY_CAPS_LOCK},
        {XK_Num_Lock, KEY_NUM_LOCK}, {XK_Scroll_Lock, KEY_SCROLL_LOCK}, {XK_Menu, KEY_MENU},
        {XK_Shift_L, KEY_SHIFT}, {XK_Shift_R, KEY_SHIFT}, {XK_Control_L, KEY_CONTROL},
        {XK_Control_R, KEY_CONTROL}, {XK_Alt_L, KEY_ALT}, {XK_Alt_R, KEY_ALT}
    };
    
    for (size_t i = 0; i < sizeof(keys)/sizeof(keys[0]); i++) {
        KeyCodeLookupTable[XKeysymToKeycode(display, keys[i].sym)] = keys[i].code;
    }

    // Function keys
    for (int i = 1; i <= 12; i++) {
        KeyCodeLookupTable[XKeysymToKeycode(display, XK_F1 + (i-1))] = KEY_F1 + (i-1);
    }

    // Numpad keys
    for (int i = 0; i <= 9; i++) {
        KeyCodeLookupTable[XKeysymToKeycode(display, XK_KP_0 + i)] = KEY_NUMPAD_0 + i;
    }
}

void platform_set_mouse_position(size_t x, size_t y) {
    XWarpPointer(display, None, window, 0, 0, 0, 0, (int)x, (int)y);
    XFlush(display);
}

void platform_enable_fullscreen() {
    Atom atoms[2] = { XInternAtom(display, "_NET_WM_STATE_FULLSCREEN", False), None };
    XChangeProperty(display, window, XInternAtom(display, "_NET_WM_STATE", False),
                   XA_ATOM, 32, PropModeReplace, (unsigned char*)atoms, 1);
    XFlush(display);
}

void platform_disable_fullscreen() {
    Atom atoms[2] = { XInternAtom(display, "_NET_WM_STATE_FULLSCREEN", False), None };
    XChangeProperty(display, window, XInternAtom(display, "_NET_WM_STATE", False),
                   XA_ATOM, 32, PropModeReplace, (unsigned char*)atoms, 0);
    XFlush(display);
}

bool platform_drag_and_drop_available() { return dropped_file_count > 0; }
char** platform_get_drag_and_drop_files(int* count) { *count = dropped_file_count; return dropped_files; }

void platform_release_drag_and_drop(char** files, int count) {
    for (int i = 0; i < dropped_file_count; i++) free((void*)dropped_files[i]);
    free(dropped_files);
    dropped_files = NULL;
    dropped_file_count = 0;
}
