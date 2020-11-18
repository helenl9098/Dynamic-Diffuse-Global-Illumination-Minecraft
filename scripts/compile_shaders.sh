#!/bin/bash
for filename in *.{vert,frag,tesc,tese,geom,comp}; do
    [ -e "$filename" ] || continue
    glslangValidator -V "$filename" -o "$filename.spv" -I"."
done
