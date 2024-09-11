#!/bin/bash

TOKEN=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | head -c 20)

gp preview $(gp url 8888)/lab?token=$TOKEN --external

jupyter lab \
    --LabApp.allow_origin=\'$(gp url 8888)\' \
    --NotebookApp.token=$TOKEN \
    --no-browser
