#!/bin/sh

if [ -n "$USE_DEBUG" ]; then

# pip install -e all plugins
# Those are folder starting with napari-*
# If not in debug mode, this will not be executed as the plugins are already installed in the Dockerfile
for plugin in /app/napari-*; do
    if [ -d "$plugin" ]; then
        echo "Installing plugin: $plugin"
        pip install -qq -e "$plugin"
    else
        echo "Skipping non-directory: $plugin"
    fi
done

fi


# check if $ USE_WEB is set and not empty 
if [ -n "$USE_WEB" ]; then
    xpra start\
        --bind-tcp=0.0.0.0:$XPRA_PORT \
        --html=on \
        --start="$XPRA_START" \
        --exit-with-client="$XPRA_EXIT_WITH_CLIENT" \
        --daemon=no \
        --xvfb="/usr/bin/Xvfb +extension Composite -screen 0 $XPRA_XVFB_SCREEN -nolisten tcp -noreset" \
        --pulseaudio=no \
        --notifications=no \
        --bell=no \
        $DISPLAY
else
    # If startup.py exists, run it
    if [ -f /app/startup.py ]; then
        python /app/startup.py
    else
        echo "No startup.py found, starting napari..."
        napari
    fi
fi
#ipython --InteractiveShellApp.exec_files="/app/startup.py"
   # --TerminalIPythonApp.exec_lines="%load_ext autoreload" \
   # --TerminalIPythonApp.exec_lines="%autoreload 2" \