#!/bin/bash

rosipython () {
    ARGS=""
    if [[ $2 ]]; then
        ARGS="-i install/$1/lib/$1/$2"
    fi
    PYTHONPATH="$PYTHONPATH:install/$1/lib/$1" ipython3 $ARGS
}
